"""Zip a finished run directory for transfer between machines."""

import hashlib
import json
import os
import zipfile

__all__ = ['zip_run', 'verify_zip']


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def zip_run(run_path, output_path=None, include_work=False):
    """Compress a run directory into a single .zip file.

    ``evaluations/iterations/`` holds hard links to call-folder files; the
    archive stores both trees as regular files.

    Parameters
    ----------
    run_path : str
        Path to the run directory (e.g. ``cases/<case>/output/<run_id>``).
    output_path : str, optional
        Output zip path. Defaults to ``<run_path>.zip``.
    include_work : bool
        Include ``su2_workdir/`` (regenerable solver CWD). Default False.

    Returns
    -------
    dict
        ``{'zip_path': ..., 'compressed_bytes': ..., 'uncompressed_bytes': ...,
           'n_files': ...}``
    """
    run_path = os.path.abspath(run_path)
    if not os.path.isdir(run_path):
        raise FileNotFoundError(f'Run directory not found: {run_path}')

    if output_path is None:
        output_path = run_path.rstrip(os.sep) + '.zip'

    run_name = os.path.basename(run_path)

    def _should_include(relpath):
        parts = relpath.split(os.sep)
        if parts[0] == 'su2_workdir' and not include_work:
            return False
        if '__pycache__' in parts:
            return False
        return True

    n_files = 0
    uncompressed = 0
    manifest = []
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirnames, filenames in os.walk(run_path):
            dirnames[:] = [d for d in dirnames
                           if d != '__pycache__'
                           and not os.path.islink(os.path.join(dirpath, d))]
            for fname in sorted(filenames):
                full = os.path.join(dirpath, fname)
                if os.path.islink(full):
                    continue
                relpath = os.path.relpath(full, run_path)
                if not _should_include(relpath):
                    continue
                arcname = os.path.join(run_name, relpath)
                zf.write(full, arcname)
                manifest.append({'path': arcname, 'sha256': _sha256(full),
                                 'bytes': os.path.getsize(full)})
                n_files += 1
                uncompressed += os.path.getsize(full)
        zf.writestr(f'{run_name}/MANIFEST.json',
                    json.dumps({'schema_version': 1, 'files': manifest},
                               indent=2) + '\n')

    compressed = os.path.getsize(output_path)
    return {
        'zip_path': output_path,
        'compressed_bytes': compressed,
        'uncompressed_bytes': uncompressed,
        'n_files': n_files,
    }


def verify_zip(zip_path):
    """Verify every archived file against the archive manifest."""
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        manifests = [name for name in names if name.endswith('/MANIFEST.json')]
        if len(manifests) != 1:
            raise ValueError(f'Archive has {len(manifests)} manifest files')
        manifest = json.loads(zf.read(manifests[0]))
        for entry in manifest.get('files', ()):
            path = entry['path']
            if path not in names:
                raise ValueError(f'Manifest entry missing from archive: {path}')
            data = zf.read(path)
            digest = hashlib.sha256(data).hexdigest()
            if digest != entry['sha256'] or len(data) != entry['bytes']:
                raise ValueError(f'Archive verification failed: {path}')
    return {'zip_path': os.path.abspath(zip_path),
            'n_files': len(manifest.get('files', ())), 'verified': True}
