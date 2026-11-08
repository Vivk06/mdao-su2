#!/usr/bin/env python
"""Zip a run directory or verify an existing archive.

The default archive path is ``<run_path>.zip``. ``--verify-only`` takes the
archive path and checks its manifest without creating another archive.

Usage::

    python run/archive.py <run_path> [--output <zip>] [--include-work]
    python run/archive.py --verify-only <zip_path>

Example::

    python run/archive.py cases/rae2822_rans_sa_hh/output/20260817T0636_sp_check_totals
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description='Zip a run directory or verify an existing archive.')
    parser.add_argument('run_path', nargs='?', help='Path to the run directory')
    parser.add_argument('--run', dest='run_option', default=None,
                        help='Path to the run directory')
    parser.add_argument('--verify-only', action='store_true',
                        help='Verify an existing zip archive')
    parser.add_argument('--output', '-o', default=None,
                        help='Output zip path (default: <run_path>.zip)')
    parser.add_argument('--include-work', action='store_true',
                        help='Include su2_workdir/ (the regenerable solver working directory)')
    args = parser.parse_args()

    from mdao_su2.run_output.archive import zip_run
    from mdao_su2.run_output.archive import verify_zip

    run_path = args.run_path or args.run_option
    if not run_path:
        parser.error('a run path is required')
    if args.verify_only:
        zip_path = args.output or run_path
        result = verify_zip(zip_path)
        print(f"  Verified: {result['zip_path']}")
        print(f"  Files: {result['n_files']}")
        return

    result = zip_run(run_path, output_path=args.output,
                     include_work=args.include_work)

    compressed_mb = result['compressed_bytes'] / (1024 * 1024)
    uncompressed_mb = result['uncompressed_bytes'] / (1024 * 1024)
    ratio = (result['compressed_bytes'] / result['uncompressed_bytes'] * 100
             if result['uncompressed_bytes'] > 0 else 0)

    print(f"  Zip: {result['zip_path']}")
    print(f"  Files: {result['n_files']}")
    print(f"  Size: {compressed_mb:.1f} MB "
          f"({uncompressed_mb:.1f} MB uncompressed, {ratio:.0f}%)")


if __name__ == '__main__':
    main()
