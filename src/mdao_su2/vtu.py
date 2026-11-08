"""Read SU2's VTU files without VTK/pyvista/meshio. VTU Float32 data is for visualisation only — use call_summary.json for numbers."""
import base64
import csv
import re
import struct
import xml.etree.ElementTree as ET
from xml.sax.saxutils import quoteattr

import numpy as np

__all__ = [
    'read_su2_vtu', 'triangles',
    'write_polydata', 'write_unstructured', 'extract_mesh_from_vtu',
    'vtu_to_csv',
    'VtuFormatError',
]

_DT = {
    'Float32': '<f4',
    'Float64': '<f8',
    'Int32': '<i4',
    'Int64': '<i8',
    'UInt8': '<u1',
    'UInt32': '<u4',
    'UInt64': '<u8',
}

_HEADER_BYTES = {
    'UInt64': 8,
    'UInt32': 4,
}


class VtuFormatError(Exception):
    """Raised when a VTU file cannot be parsed."""


def _parse_xml_header(raw):
    """Return (ET root, appended bytes or None, header_type str)."""
    appended = None
    appended_idx = raw.find(b'<AppendedData')
    if appended_idx >= 0:
        for marker in [b'<AppendedData encoding="raw">\n_',
                       b'<AppendedData encoding="raw">\r\n_']:
            idx = raw.find(marker)
            if idx >= 0:
                appended = raw[idx + len(marker):]
                break
        xml_bytes = raw[:appended_idx]
    else:
        xml_bytes = raw

    xml_text = xml_bytes.decode('utf-8', errors='replace').strip()
    if not xml_text.rstrip().endswith('</VTKFile>'):
        xml_text += '\n</VTKFile>'
    xml_text = re.sub(r'(\w+)=\s+"', r'\1="', xml_text)
    root = ET.fromstring(xml_text)

    header_type = root.attrib.get('header_type', 'UInt32')
    if header_type not in _HEADER_BYTES:
        raise VtuFormatError(
            f'Unsupported header_type={header_type!r}; '
            f'expected one of {list(_HEADER_BYTES)}')
    return root, appended, header_type


def _read_array_appended(da, n, appended, hdr_bytes):
    """Read a raw-appended DataArray."""
    if appended is None:
        raise VtuFormatError(
            'DataArray uses appended format but no '
            '<AppendedData encoding="raw"> block found in file')
    dtype = _DT.get(da.attrib['type'])
    if dtype is None:
        raise VtuFormatError(
            f'Unsupported data type {da.attrib["type"]!r}; '
            f'supported: {list(_DT)}')
    nc = int(da.attrib.get('NumberOfComponents', '1'))
    off = int(da.attrib['offset'])
    a = np.frombuffer(appended, dtype=dtype, count=n * nc,
                      offset=off + hdr_bytes).copy()
    return a.reshape(n, nc) if nc > 1 else a


def _read_array_inline(da, n):
    """Read an ascii or base64 inline DataArray."""
    dtype = _DT.get(da.attrib['type'])
    if dtype is None:
        raise VtuFormatError(
            f'Unsupported data type {da.attrib["type"]!r}; '
            f'supported: {list(_DT)}')
    nc = int(da.attrib.get('NumberOfComponents', '1'))
    fmt = da.attrib.get('format', 'ascii')
    text = da.text
    if text is None:
        raise VtuFormatError(
            f'Inline DataArray {da.attrib.get("Name", "?")} has no text')

    if fmt == 'ascii':
        a = np.fromstring(text.strip(), dtype=dtype, sep=' ')
    elif fmt == 'binary':
        decoded = base64.b64decode(text.strip())
        hdr_type = da.attrib.get('header_type', 'UInt32')
        hdr_bytes = _HEADER_BYTES.get(hdr_type, 4)
        a = np.frombuffer(decoded, dtype=dtype, offset=hdr_bytes).copy()
    else:
        raise VtuFormatError(f'Unsupported DataArray format={fmt!r}')

    expected = n * nc
    if a.size < expected:
        raise VtuFormatError(
            f'DataArray {da.attrib.get("Name", "?")} has {a.size} values, '
            f'expected {expected}')
    a = a[:expected]
    return a.reshape(n, nc) if nc > 1 else a


def _read_array(da, n, appended, hdr_bytes):
    """Dispatch to appended or inline reader."""
    fmt = da.attrib.get('format', 'ascii')
    if fmt == 'appended':
        return _read_array_appended(da, n, appended, hdr_bytes)
    return _read_array_inline(da, n)


def read_su2_vtu(path):
    """Read a VTU file into {points, point_data, cell_data, cell_types, cells}."""
    with open(path, 'rb') as f:
        raw = f.read()

    root, appended, header_type = _parse_xml_header(raw)
    hdr_bytes = _HEADER_BYTES[header_type]

    pieces = root.findall('.//Piece')
    if not pieces:
        raise VtuFormatError('No <Piece> element found')

    all_points = []
    all_point_data = {}
    all_cell_data = {}
    all_cell_types = []
    all_cells = []
    point_offset = 0

    for piece in pieces:
        npts = int(piece.attrib['NumberOfPoints'])
        ncells = int(piece.attrib['NumberOfCells'])

        pts = _read_array(piece.find('Points/DataArray'), npts,
                          appended, hdr_bytes)
        all_points.append(pts)

        for da in piece.findall('PointData/DataArray'):
            name = da.attrib['Name']
            arr = _read_array(da, npts, appended, hdr_bytes)
            if name in all_point_data:
                all_point_data[name] = np.concatenate(
                    [all_point_data[name], arr])
            else:
                all_point_data[name] = arr

        for da in piece.findall('CellData/DataArray'):
            name = da.attrib['Name']
            arr = _read_array(da, ncells, appended, hdr_bytes)
            if name in all_cell_data:
                all_cell_data[name] = np.concatenate(
                    [all_cell_data[name], arr])
            else:
                all_cell_data[name] = arr

        cells_el = piece.find('Cells')
        if cells_el is not None and ncells > 0:
            by_name = {da.attrib['Name']: da
                       for da in cells_el.findall('DataArray')}
            offsets = _read_array(by_name['offsets'], ncells,
                                 appended, hdr_bytes)
            conn = _read_array(by_name['connectivity'], int(offsets[-1]),
                               appended, hdr_bytes)
            types = _read_array(by_name['types'], ncells,
                                appended, hdr_bytes)
            all_cell_types.append(types)
            starts = np.concatenate([[0], offsets[:-1]])
            for a, b in zip(starts, offsets):
                cell = conn[int(a):int(b)]
                all_cells.append(cell + point_offset)

        point_offset += npts

    result = {
        'points': np.concatenate(all_points) if len(all_points) > 1
                  else all_points[0],
        'point_data': all_point_data,
        'cell_data': all_cell_data,
    }
    if all_cell_types:
        result['cell_types'] = np.concatenate(all_cell_types) \
            if len(all_cell_types) > 1 else all_cell_types[0]
        result['cells'] = all_cells
    return result


def triangles(surface):
    """``(ntri, 3)`` vertex indices, fan-triangulating quads for matplotlib."""
    tri = []
    for c in surface['cells']:
        for t in range(1, len(c) - 1):
            tri.append((c[0], c[t], c[t + 1]))
    return np.asarray(tri, dtype=int)


def _fmt_array(arr, nc):
    """Format array values for ASCII VTK XML output."""
    flat = arr.ravel()
    lines = []
    per_line = max(nc, 6)
    fmt = (lambda v: str(int(v))) if flat.dtype.kind in 'iu' else (lambda v: f'{v:.10g}')
    for i in range(0, len(flat), per_line):
        chunk = flat[i:i + per_line]
        lines.append(' '.join(fmt(v) for v in chunk))
    return '\n'.join(lines)


def _typed_array(name, arr):
    """``<DataArray>`` lines of one named array; integers stay Int32 so ParaView
    treats them as ids rather than a continuous field."""
    arr = np.asarray(arr)
    nc = arr.shape[1] if arr.ndim == 2 else 1
    kind = 'Int32' if arr.dtype.kind in 'iu' else 'Float64'
    return [f'<DataArray type="{kind}" Name={quoteattr(str(name))} '
            f'NumberOfComponents="{nc}" format="ascii">',
            _fmt_array(arr, nc),
            '</DataArray>']


def _write_vtk_xml(path, vtk_type, points, connectivity_block,
                   point_data=None, n_cells=0, cell_data=None, field_data=None):
    """Write an ASCII VTK XML file.

    ``cell_data`` maps a name to one value per cell. ``field_data`` maps a name
    to a small array that belongs to the whole file (for example a legend).
    """
    npts = len(points)
    nc_pts = points.shape[1] if points.ndim == 2 else 3
    lines = [
        '<?xml version="1.0"?>',
        f'<VTKFile type="{vtk_type}" version="1.0" '
        f'byte_order="LittleEndian" header_type="UInt64">',
        f'<{vtk_type}>',
    ]
    if field_data:
        lines.append('<FieldData>')
        for name, arr in field_data.items():
            lines.extend(_typed_array(name, np.atleast_1d(arr)))
        lines.append('</FieldData>')
    lines += [
        f'<Piece NumberOfPoints="{npts}" NumberOfCells="{n_cells}">',
        '<Points>',
        f'<DataArray type="Float64" NumberOfComponents="{nc_pts}" '
        f'format="ascii">',
        _fmt_array(points, nc_pts),
        '</DataArray>',
        '</Points>',
    ]
    if point_data:
        lines.append('<PointData>')
        for name, arr in point_data.items():
            arr = np.asarray(arr)
            nc = arr.shape[1] if arr.ndim == 2 else 1
            lines.append(
                f'<DataArray type="Float64" Name="{name}" '
                f'NumberOfComponents="{nc}" format="ascii">')
            lines.append(_fmt_array(arr, nc))
            lines.append('</DataArray>')
        lines.append('</PointData>')
    if cell_data:
        lines.append('<CellData>')
        for name, arr in cell_data.items():
            lines.extend(_typed_array(name, arr))
        lines.append('</CellData>')

    lines.append(connectivity_block)

    lines.extend([
        '</Piece>',
        f'</{vtk_type}>',
        '</VTKFile>',
    ])
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def write_polydata(path, points, lines_conn, point_data=None):
    """Write an ASCII VTP (PolyData) file."""
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(-1, 3)
    if points.shape[1] == 2:
        points = np.column_stack([points, np.zeros(len(points))])

    n_lines = len(lines_conn)
    conn_parts = []
    offsets = []
    running = 0
    for seg in lines_conn:
        seg = np.asarray(seg)
        running += len(seg)
        offsets.append(running)
        conn_parts.append(' '.join(str(int(v)) for v in seg))

    block = ['<Lines>',
             '<DataArray type="Int64" Name="connectivity" format="ascii">',
             '\n'.join(conn_parts),
             '</DataArray>',
             '<DataArray type="Int64" Name="offsets" format="ascii">',
             ' '.join(str(o) for o in offsets),
             '</DataArray>',
             '</Lines>']

    _write_vtk_xml(path, 'PolyData', points, '\n'.join(block),
                   point_data, n_cells=n_lines)


def write_unstructured(path, points, cells, cell_types, point_data=None,
                       cell_data=None, field_data=None):
    """Write an ASCII VTU (UnstructuredGrid) file."""
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(-1, 3)
    if points.shape[1] == 2:
        points = np.column_stack([points, np.zeros(len(points))])

    ncells = len(cells)
    conn_parts = []
    offsets = []
    running = 0
    for cell in cells:
        cell = np.asarray(cell)
        running += len(cell)
        offsets.append(running)
        conn_parts.append(' '.join(str(int(v)) for v in cell))

    cell_types = np.asarray(cell_types, dtype=int)
    block = ['<Cells>',
             '<DataArray type="Int64" Name="connectivity" format="ascii">',
             '\n'.join(conn_parts),
             '</DataArray>',
             '<DataArray type="Int64" Name="offsets" format="ascii">',
             ' '.join(str(o) for o in offsets),
             '</DataArray>',
             '<DataArray type="Int64" Name="types" format="ascii">',
             ' '.join(str(int(t)) for t in cell_types),
             '</DataArray>',
             '</Cells>']

    _write_vtk_xml(path, 'UnstructuredGrid', points, '\n'.join(block),
                   point_data, n_cells=ncells, cell_data=cell_data,
                   field_data=field_data)


def extract_mesh_from_vtu(src_vtu, dst_vtu):
    """Read a flow/adjoint VTU and write back mesh geometry only (no field data)."""
    data = read_su2_vtu(src_vtu)
    write_unstructured(dst_vtu, data['points'], data['cells'],
                       data['cell_types'])


def vtu_to_csv(src_vtu, dst_csv):
    """Export a VTU's coordinates and data arrays to a deterministic CSV.

    Point and cell arrays have different cardinalities.  A single CSV therefore
    uses row number as their common index and leaves the shorter kind blank.
    This keeps every value while retaining the useful point-wise table shape.
    """
    data = read_su2_vtu(src_vtu)  # deliberately raises on a bad copied VTU
    points = np.asarray(data['points'])

    def columns(prefix, arrays):
        out = []
        for name in sorted(arrays):
            a = np.asarray(arrays[name])
            a = a.reshape(-1, 1) if a.ndim == 1 else a.reshape(len(a), -1)
            suffixes = ['' if a.shape[1] == 1 else f'_{i}'
                        for i in range(a.shape[1])]
            out.extend((f'{prefix}{name}{suffix}', a[:, i])
                       for i, suffix in enumerate(suffixes))
        return out

    pcols = columns('', data['point_data'])
    ccols = columns('cell_', data['cell_data'])
    nrows = max([len(points)] + [len(a) for _, a in pcols + ccols])
    headers = ['x', 'y', 'z'] + [n for n, _ in pcols + ccols]
    with open(dst_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i in range(nrows):
            row = [points[i, j] if i < len(points) and j < points.shape[1]
                   else '' for j in range(3)]
            row.extend(a[i] if i < len(a) else '' for _, a in pcols + ccols)
            writer.writerow(row)
