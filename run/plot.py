#!/usr/bin/env python
"""Plot run data or convert an SU2 mesh to VTU.

Plot commands take explicit input files and write figures to ``-o/--out DIR``.
Use each subcommand's help for its options. A leading minus in a value is
written with ``=`` (for example, ``--xlim=-0.1,1.1``).

Exit status: 0 done, 2 usage or configuration error, 1 failed.
"""

import argparse
import os
import sys

os.environ.setdefault('OPENMDAO_REPORTS', '0')


class UsageError(Exception):
    """A command-line value that cannot be used (exit 2)."""


# ---- value parsers -------------------------------------------------------------------

def _list(value):
    """``'a,b'`` as ``['a', 'b']``, or None."""
    items = [v.strip() for v in value.split(',') if v.strip()] if value else []
    return items or None


def _floats(value, what):
    try:
        return [float(v) for v in _list(value) or []] or None
    except ValueError:
        raise UsageError(f'{what} takes comma-separated numbers, got {value!r}') from None


def _pair(value, what):
    numbers = _floats(value, what)
    if numbers is None or len(numbers) != 2:
        raise UsageError(f'{what} takes two numbers, lo,hi')
    return tuple(numbers)


def _coord(value, what):
    numbers = _floats(value, what)
    if not numbers:
        raise UsageError(f'{what} takes comma-separated coordinates')
    return tuple(numbers)


def _scalar_or_list(numbers):
    return numbers[0] if numbers and len(numbers) == 1 else numbers


def _labels(args, count):
    labels = args.label or None
    if labels and len(labels) != count:
        raise UsageError(f'{count} inputs but {len(labels)} --label values')
    return labels


# ---- argument groups --------------------------------------------------------------

def _shared(p):
    g = p.add_argument_group('output and style')
    g.add_argument('-o', '--out', required=True, metavar='DIR', help='output directory (created)')
    g.add_argument('--format', dest='fmt', choices=['png', 'pdf'], default='png',
                   help='image format (default png)')
    g.add_argument('--dpi', type=int, help='resolution (default the package DPI)')
    g.add_argument('--figsize', metavar='WxH',
                   help='figure size in inches, e.g. 8x4 (default the figure\'s own)')
    g.add_argument('--colors', metavar='LIST', help='comma-separated colours, one per entry')
    g.add_argument('--styles', metavar='LIST',
                   help='comma-separated linestyles, one per entry; write --styles=-,--')
    g.add_argument('--palette', choices=['default', 'categorical'], default='default',
                   help='colour sequence when --colors is not given')
    g.add_argument('--legend-out', action='store_true', help='also write legend.<fmt>')
    g.add_argument('--xlim', metavar='LO,HI', help='x limits of every panel; write --xlim=LO,HI')
    g.add_argument('--ylim', metavar='LO,HI',
                   help='y limits of a one-panel figure; write --ylim=LO,HI')
    g.add_argument('--figures', metavar='LIST', help='comma-separated figure names (default set)')


def _inputs(p, what):
    p.add_argument('files', nargs='*', metavar='FILE', help=f'{what} (N files = N entries)')
    p.add_argument('--file', action='append', default=[], metavar='FILE', dest='more_files',
                   help='one more input file (repeatable)')
    p.add_argument('--label', action='append', default=[], metavar='TEXT',
                   help='entry label, repeated, order-matched to the files (default file stem)')


def _physics(p):
    g = p.add_argument_group('freestream scalars (surface_mach, span_load)')
    g.add_argument('--mach', metavar='M[,M...]', help='Mach: one value or one per file')
    g.add_argument('--gamma', metavar='G', help='ratio of specific heats')
    g.add_argument('--cfg', action='append', default=[], metavar='FILE',
                   help='SU2 cfg giving MACH_NUMBER and GAMMA_VALUE (surface_mach) or AOA and REF_AREA (span_load): one, or one per file')


class _Group(argparse.Action):
    """``--curve LABEL`` (or ``--reference LABEL``) starts a sample group; ``--sample`` joins it."""

    def __call__(self, parser, namespace, values, option_string=None):
        groups = namespace.groups
        if self.dest in ('curve', 'reference'):
            groups.append({'label': values, 'samples': [],
                           'reference': self.dest == 'reference'})
            return
        if not groups:
            groups.append({'label': None, 'samples': [], 'reference': False})
        groups[-1]['samples'].append(values)


def _groups(p, what):
    p.add_argument('--curve', action=_Group, default=None, metavar='LABEL', dest='curve',
                   help=f'start a new {what} group named LABEL; later --sample belong to it')
    p.add_argument('--sample', action=_Group, default=None, metavar='FILE[:K=V,...]',
                   dest='sample',
                   help='a sample file with optional NAME=VALUE overrides (VALUE a number or '
                        'an expression over the sample), and row=N, cfg=PATH, point=ID, '
                        'kind=history|hst|stdout|polar|table; FILE may join files with +')
    p.add_argument('--cfg', metavar='FILE',
                   help='SU2 cfg whose numeric keys (MACH, GAMMA, AOA...) join every sample')
    p.add_argument('--row', type=int, default=-1,
                   help='history/stdout row or .hst major iteration of every sample (default -1)')
    p.set_defaults(groups=[])


def _sample_specs(args):
    """``[(label, [SampleSpec])]`` of the ``--curve``/``--sample`` flags."""
    from mdao_su2.postprocess.series import SampleSpec
    groups = []
    for group in args.groups:
        specs = []
        for text in group['samples']:
            head, sep, tail = text.rpartition(':')
            items = tail.split(',') if sep else []
            if sep and all('=' in i and i.partition('=')[0].strip().isidentifier() for i in items):
                files = head
            else:
                files, items = text, []
            options = {'row': args.row, 'cfg': args.cfg, 'kind': None, 'point': None}
            overrides = []
            for item in items:
                key, _, val = item.partition('=')
                key = key.strip()
                if key in ('cfg', 'kind', 'point'):
                    options[key] = val
                elif key == 'row':
                    try:
                        options['row'] = int(val)
                    except ValueError:
                        raise UsageError(f'row={val!r} in --sample {text!r} is not an integer') \
                            from None
                else:
                    overrides.append(item)
            for path in files.split('+'):
                if not path:
                    raise UsageError(f'empty file name in --sample {text!r}')
                specs.append(SampleSpec(path=path, row=options['row'], kind=options['kind'],
                                        cfg=options['cfg'], overrides=tuple(overrides),
                                        point=options['point']))
        if not specs:
            raise UsageError('a --curve has no --sample')
        groups.append((group['label'], specs, group.get('reference', False)))
    if not groups:
        raise UsageError('give at least one --sample')
    return groups


def _group_labels(args, groups):
    labels = args.label or None
    if labels:
        if len(labels) != len(groups):
            raise UsageError(f'{len(groups)} curves but {len(labels)} --label values')
        return labels
    named = [g[0] for g in groups]
    return [n or f'series {i + 1}' for i, n in enumerate(named)]


# ---- parser ---------------------------------------------------------------------------

def build_parser():
    root = argparse.ArgumentParser(
        prog='run/plot.py', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    root.epilog = (
        'Surfaces and fields\n'
        '  surface    Cp, field, Mach and wing figures\n'
        '  shape      wall outline and displacement\n'
        '  view       isometric 3D surface view\n'
        '  field      volume contours\n'
        '  probe      sample a volume\n'
        'Convergence\n'
        '  history    SU2 history figures\n'
        '  stdout     SU2 screen output figures\n'
        'Optimization\n'
        '  optimizer  optimizer history figures\n'
        '  sweep      compare sample curves\n'
        '  iterations compare iteration samples\n'
        'Design variables\n'
        '  dv-map     geometry and basis figures\n'
        'Conversion\n'
        '  convert    SU2 mesh to volume and marker VTU files\n\n'
        'Examples using kept cases:\n'
        '  python run/plot.py convert '
        'cases/adodg1_naca0012_euler_ffd/input/mesh/adodg1_euler_full129.su2 '
        '-o /tmp/mdao-su2-mesh2d.vtu\n'
        '  python run/plot.py convert '
        'cases/oneram6_euler_ffd/input/mesh/mesh_ONERAM6_inv_FFD.su2 '
        '-o /tmp/mdao-su2-mesh3d.vtu\n'
        '  python run/plot.py dv-map --config '
        'cases/adodg1_naca0012_euler_ffd/input/configs/opt_ffd_cp.toml '
        '-o /tmp/mdao-su2-figures/')
    sub = root.add_subparsers(dest='command', metavar='COMMAND', required=True)

    def add(name, summary):
        p = sub.add_parser(name, help=summary, description=summary)
        _shared(p)
        return p

    p = add('surface', 'Cp, field, Mach and wing figures of N surface files')
    _inputs(p, 'surface .vtu or SU2 surface .csv')
    p.add_argument('--reference', action='append', default=[], metavar='FILE[:point=N]',
                   help='reference Cp CSV drawn as markers on cp_chord (repeatable); '
                        'point=N picks one point of a multi-point file')
    p.add_argument('--reference-label', action='append', default=[], metavar='TEXT',
                   help='reference label, order-matched (default file stem [@point N])')
    p.add_argument('--field', metavar='LIST', help='point-data fields (default Pressure_Coefficient)')
    p.add_argument('--stations', metavar='LIST', help='span stations of panel_sections')
    p.add_argument('--span-axis', default='y', choices=['x', 'y', 'z'], help='wing span axis')
    p.add_argument('--lift-axis', default='z', choices=['x', 'y', 'z'], help='wing lift axis')
    p.add_argument('--aoa', metavar='A[,A...]',
                   help='angle of attack in degrees for span_load: one value or one per file '
                        '(else AOA of --cfg)')
    p.add_argument('--ref-area', metavar='S[,S...]',
                   help='reference area of the modelled wing for span_load (else REF_AREA of --cfg)')
    p.add_argument('--semispan', type=float, metavar='B',
                   help='semispan coordinate extent for span_load')
    p.add_argument('--n-stations', type=int, default=60, help='span_load strips (default 60)')
    p.add_argument('--quantities', metavar='LIST',
                   help='panels of section_geometry: twist, chord, thickness (default all three)')
    p.add_argument('--displacement-scale', type=float, default=1.0, metavar='K',
                   help='surface_3d_overlay: draw each later surface at first + K x its '
                        'displacement (default 1; the legend states K when not 1)')
    _physics(p)

    p = add('shape', 'wall outline and displacement of N 2D surface files (first is the base)')
    _inputs(p, 'surface .vtu or .csv')

    p = add('view', 'isometric view of one 3D surface file')
    p.add_argument('surface', metavar='SURFACE', help='surface .vtu')
    p.add_argument('--baseline', metavar='FILE', help='a second surface drawn faint')
    p.add_argument('--field', help='point-data field to colour by')
    p.add_argument('--lift-axis', default='z', choices=['x', 'y', 'z'], help='wing lift axis')

    p = add('field', 'contour of N volume files on one colour scale per field')
    _inputs(p, 'volume .vtu')
    p.add_argument('--field', metavar='LIST', default='Mach', help='fields (default Mach)')
    p.add_argument('--wall', action='append', default=[], metavar='FILE',
                   help='surface file of each volume (repeat, one per volume): sets the window')
    p.add_argument('--length', type=float, default=1.0, help='divide coordinates by this')
    p.add_argument('--margin', type=float, default=0.3, help='window margin in chords')
    p.add_argument('--levels', type=int, default=28, help='contour levels')

    for name, what in (('history', 'SU2 history .csv or tecplot .dat'),
                       ('stdout', 'SU2 screen output .out/.log')):
        p = add(name, f'residual and coefficient figures of N {what} files')
        _inputs(p, what)
        p.add_argument('--columns', metavar='LIST',
                       help='comma-separated expressions over the columns; a name with brackets is '
                            'written by its alias, e.g. rms_Rho,CL/CD')
        p.add_argument('--x', metavar='EXPR', help='x-axis expression (default the row index)')
        if name == 'history':
            p.add_argument('--floor', type=float, help='residual floor of adjoint_residuals')
            p.add_argument('--floor-cfg', metavar='FILE', help='SU2 cfg giving CONV_RESIDUAL_MINVAL')
            p.add_argument('--adjoint-column', default='rms[A_Rho]',
                           help='column of adjoint_residuals (default rms[A_Rho])')

    p = add('optimizer', 'objective, constraint and design-variable figures of N .hst files')
    _inputs(p, 'pyOptSparse .hst')
    p.add_argument('--constraints', metavar='LIST',
                   help='comma-separated NAME[=TARGET][:eq|ineq] (default every scalar constraint)')
    p.add_argument('--dv', metavar='NAME', help='design variable of dv_profile')
    p.add_argument('--crest', metavar='LIST', help='bump crest positions (draws dv_profile)')
    p.add_argument('--n-dv', type=int, help='Hicks-Henne bumps per surface: crests from --distribution')
    p.add_argument('--distribution', choices=['even', '1-cos'], help='bump distribution')
    p.add_argument('--bounds', metavar='LO,HI', help='design-variable bounds; write --bounds=LO,HI')
    p.add_argument('--unit', type=float, default=1e-3, help='dv_profile unit (default 1e-3)')
    p.add_argument('--iteration', type=int, default=-1, help='major iteration of dv_profile')

    p = add('sweep', 'expressions of sample files against an x expression, N curves')
    _groups(p, 'curve')
    p.add_argument('--reference', action=_Group, default=None, metavar='LABEL',
                   dest='reference',
                   help='start a reference group named LABEL, drawn as markers with the '
                        'comparison E = S - D against every curve; later --sample belong to it')
    p.add_argument('--label', action='append', default=[], metavar='TEXT',
                   help='curve label (alternative to --curve LABEL), order-matched over all '
                        'groups including references')
    p.add_argument('--x', required=True, metavar='EXPR', help='x expression over the samples')
    p.add_argument('--y', action='append', required=True, metavar='EXPR',
                   help='y expression, repeat for one panel each')
    p.add_argument('--peak', action='append', default=[], nargs='?', const='', metavar='EXPR',
                   help='peak marker on a y expression (bare --peak: every y)')
    p.add_argument('--log-y', action='append', default=[], nargs='?', const='', metavar='EXPR',
                   help='log axis for a y expression (bare --log-y: every y)')
    p.add_argument('--x-label', help='x axis label (default the expression)')
    p.add_argument('--y-label', action='append', default=[], metavar='TEXT',
                   help='y axis label, one per --y in order')

    p = add('iterations', 'per-iteration change and linear-constraint margins, N groups')
    _groups(p, 'group')
    p.add_argument('--label', action='append', default=[], metavar='TEXT',
                   help='group label (alternative to --curve LABEL), order-matched')
    p.add_argument('--expr', action='append', default=[], metavar='EXPR',
                   help='expression whose change from iteration 0 is drawn, one panel each')
    p.add_argument('--margin', action='append', default=[], metavar='EXPR>=VAL',
                   help='linear constraint margin: EXPR>=VAL, EXPR<=VAL or EXPR==VAL; '
                        'EXPR is linear in the sample names')
    p.add_argument('--absolute', action='store_true', help='draw differences, not percent')
    p.add_argument('--y-label', action='append', default=[], metavar='TEXT',
                   help='y axis label, one per --expr in order (default change from iteration 0)')

    p = add('probe', 'sample a volume file along a line, box or plane')
    p.add_argument('volume_pos', nargs='?', metavar='VOLUME', help='volume .vtu')
    p.add_argument('--volume', metavar='FILE', help='volume .vtu')
    p.add_argument('--field', metavar='LIST', help='fields to sample (default all)')
    p.add_argument('--ndim', type=int, choices=[2, 3], default=2, help='dimension (default 2)')
    p.add_argument('--probe-line', nargs=2, metavar=('P0', 'P1'), help='two comma-separated points')
    p.add_argument('--probe-box', nargs=2, metavar=('LO', 'HI'), help='two corners (2D)')
    p.add_argument('--probe-plane', metavar='ORIGIN', help='plane origin (3D)')
    p.add_argument('--normal', metavar='NX,NY,NZ', help='plane normal (with --probe-plane)')
    p.add_argument('--n', metavar='N', help='samples: N for a line, NX,NY for a region')
    p.add_argument('--name', help='probe name (default cli_<kind>)')
    p.add_argument('--data-only', action='store_true', help='write the probe data, no figure')

    p = add('dv-map', 'design-variable map and basis figures of a run config (geometry only)')
    p.add_argument('--config', required=True, metavar='TOML', help='run TOML')
    p.add_argument('--wall', metavar='FILE', help='surface file drawn with the FFD box')
    p.add_argument('--dvs', metavar='NPY', help='design variables (.npy) of the FFD influence panel')
    p = sub.add_parser('convert', help='convert an SU2 mesh to VTU',
                       description='Write volume cells and marker cells to separate VTU files.')
    p.add_argument('mesh', metavar='MESH.su2', help='2D or 3D SU2 mesh')
    p.add_argument('-o', '--out', required=True, metavar='MESH.vtu',
                   help='volume VTU path')
    p.add_argument('--marker', action='append', dest='markers', metavar='TAG',
                   help='include this boundary marker (repeatable; default all)')
    sub._choices_actions.clear()
    return root


# ---- running ---------------------------------------------------------------------------

def _options(args, argv):
    from mdao_su2.postprocess.plots import PlotOptions
    figsize = None
    if args.figsize:
        try:
            w, h = args.figsize.lower().split('x')
            figsize = (float(w), float(h))
        except ValueError:
            raise UsageError('--figsize takes WxH in inches, e.g. 8x4') from None
    return PlotOptions(
        fmt=args.fmt, dpi=args.dpi, figsize=figsize, colors=_list(args.colors),
        styles=_list(args.styles), palette=args.palette,
        xlim=_pair(args.xlim, '--xlim') if args.xlim else None,
        ylim=_pair(args.ylim, '--ylim') if args.ylim else None,
        legend_out=args.legend_out, cmd='plot.py ' + ' '.join(argv))


def _files(args):
    files = list(getattr(args, 'files', [])) + list(getattr(args, 'more_files', []))
    if not files:
        raise UsageError('give at least one input file')
    return files


def _mach_gamma(args):
    mach = _scalar_or_list(_floats(args.mach, '--mach'))
    gamma = _scalar_or_list(_floats(args.gamma, '--gamma'))
    return mach, gamma, (args.cfg or None)


def _constraints(text):
    """``{name: {'target', 'kind'}}`` of ``NAME[=TARGET][:eq|ineq]`` items."""
    if not text:
        return None
    specs = {}
    for item in _list(text):
        name, explicit = item, None
        head, sep, tail = item.rpartition(':')
        if sep and tail in ('eq', 'ineq'):
            name, explicit = head, tail
        target = None
        if '=' in name:
            name, _, value = name.partition('=')
            try:
                target = float(value)
            except ValueError:
                raise UsageError(f'--constraints target {value!r} is not a number') from None
        kind = explicit or ('eq' if target is not None else 'ineq')
        specs[name] = {'target': target, 'kind': kind}
    return specs


def _crest(args):
    if args.crest:
        return _floats(args.crest, '--crest')
    if args.n_dv or args.distribution:
        if not (args.n_dv and args.distribution):
            raise UsageError('--n-dv and --distribution go together')
        from mdao_su2.geometry.hicks_henne import compute_bump_locations
        return [float(v) for v in compute_bump_locations(args.n_dv, args.distribution)]
    return None


def _margin_rows(items):
    """Rows for ``plot_iterations`` from ``EXPR>=VAL`` / ``<=`` / ``==`` strings."""
    from mdao_su2.postprocess.series import SeriesError, linear_terms
    rows = []
    for text in items:
        for op, key in (('>=', 'lower'), ('<=', 'upper'), ('==', 'equals')):
            if op in text:
                expr, _, value = text.partition(op)
                break
        else:
            raise UsageError(f'--margin {text!r} needs >=, <= or ==')
        try:
            bound = float(value)
        except ValueError:
            raise UsageError(f'--margin {text!r}: {value!r} is not a number') from None
        try:
            terms, offset = linear_terms(expr)
        except SeriesError as exc:
            raise UsageError(f'--margin {text!r}: {exc}') from exc
        rows.append({'name': text.strip(), 'terms': terms, 'offset': offset, key: bound})
    return rows


def _probe_spec(args):
    fields = _list(args.field)

    def region_n():
        if not args.n:
            return (200, 200)
        parts = [int(v) for v in args.n.split(',')]
        return (parts[0], parts[1]) if len(parts) == 2 else (parts[0], parts[0])

    try:
        if args.probe_line:
            return {'kind': 'line', 'p0': _coord(args.probe_line[0], '--probe-line'),
                    'p1': _coord(args.probe_line[1], '--probe-line'),
                    'n': int(args.n) if args.n else 200,
                    'name': args.name or 'cli_line', 'fields': fields}
        if args.probe_box:
            return {'kind': 'box', 'lo': _coord(args.probe_box[0], '--probe-box'),
                    'hi': _coord(args.probe_box[1], '--probe-box'), 'n': region_n(),
                    'name': args.name or 'cli_box', 'fields': fields}
        if args.probe_plane:
            if not args.normal:
                raise UsageError('--probe-plane requires --normal')
            return {'kind': 'plane', 'origin': _coord(args.probe_plane, '--probe-plane'),
                    'normal': _coord(args.normal, '--normal'), 'n': region_n(),
                    'name': args.name or 'cli_plane', 'fields': fields}
    except ValueError:
        raise UsageError('--n takes integers: N, or NX,NY') from None
    raise UsageError('give one of --probe-line, --probe-box, --probe-plane')


def run(args, argv):
    if args.command == 'convert':
        from mdao_su2.mesh.su2_mesh_file import su2_mesh_to_vtu
        volume = args.out
        boundary = os.path.splitext(volume)[0] + '_markers.vtu'
        os.makedirs(os.path.dirname(os.path.abspath(volume)), exist_ok=True)
        su2_mesh_to_vtu(args.mesh, volume, boundary_path=boundary,
                        markers=args.markers)
        return {'mesh': volume, 'markers': boundary}
    from mdao_su2.postprocess import plots
    options = _options(args, argv)
    out = args.out
    figures = _list(args.figures)
    cmd = args.command

    if cmd == 'surface':
        files = _files(args)
        mach, gamma, cfgs = _mach_gamma(args)
        return plots.plot_surface(
            files, out, labels=_labels(args, len(files)), fields=_list(args.field),
            figures=figures, mach=mach, gamma=gamma, cfgs=cfgs,
            stations=_floats(args.stations, '--stations'), span_axis=args.span_axis,
            lift_axis=args.lift_axis, options=options, references=args.reference or None,
            reference_labels=args.reference_label or None,
            aoa=_scalar_or_list(_floats(args.aoa, '--aoa')),
            ref_area=_scalar_or_list(_floats(args.ref_area, '--ref-area')),
            semispan=args.semispan, n_stations=args.n_stations,
            quantities=_list(args.quantities), displacement_scale=args.displacement_scale)
    if cmd == 'shape':
        files = _files(args)
        return plots.plot_shape(files, out, labels=_labels(args, len(files)), figures=figures,
                                options=options)
    if cmd == 'view':
        return plots.plot_view(args.surface, out, field=args.field, baseline=args.baseline,
                               lift_axis=args.lift_axis, options=options)
    if cmd == 'field':
        files = _files(args)
        return plots.plot_field(
            files, out, labels=_labels(args, len(files)), fields=_list(args.field),
            walls=args.wall or None, length=args.length, margin=args.margin,
            levels=args.levels, options=options)
    if cmd in ('history', 'stdout'):
        files = _files(args)
        common = dict(labels=_labels(args, len(files)), figures=figures,
                      columns=_list(args.columns), x=args.x, options=options)
        if cmd == 'stdout':
            return plots.plot_stdout(files, out, **common)
        return plots.plot_history(files, out, floor=args.floor, floor_cfg=args.floor_cfg,
                                  adjoint_column=args.adjoint_column, **common)
    if cmd == 'optimizer':
        files = _files(args)
        bounds = _pair(args.bounds, '--bounds') if args.bounds else (None, None)
        return plots.plot_optimizer(
            files, out, labels=_labels(args, len(files)), figures=figures,
            constraints=_constraints(args.constraints), dv=args.dv, crest=_crest(args),
            bounds=bounds, unit=args.unit, iteration=args.iteration, options=options)
    if cmd == 'sweep':
        groups = _sample_specs(args)
        y = list(args.y)
        peak = y if '' in args.peak else args.peak
        log_y = y if '' in args.log_y else args.log_y
        y_labels = dict(zip(y, args.y_label)) if args.y_label else None
        return plots.plot_sweep(
            [g[1] for g in groups], out, args.x, y,
            labels=_group_labels(args, groups), peak=peak, log_y=log_y, y_labels=y_labels,
            x_label=args.x_label, options=options, reference=[g[2] for g in groups])
    if cmd == 'iterations':
        groups = _sample_specs(args)
        if not (args.expr or args.margin):
            raise UsageError('give --expr or --margin')
        return plots.plot_iterations(
            [g[1] for g in groups], out, exprs=args.expr,
            rows=_margin_rows(args.margin), labels=_group_labels(args, groups),
            percent=not args.absolute, options=options,
            y_labels=dict(zip(args.expr, args.y_label)) if args.y_label else None)
    if cmd == 'probe':
        volume = args.volume or args.volume_pos
        if not volume:
            raise UsageError('give the volume file')
        spec = _probe_spec(args)
        if args.data_only:
            metas = plots.run_probes(volume, [spec], out, ndim=args.ndim)
            return {f"probe_{m['name']}": os.path.join(out, 'probes', m['name']) for m in metas}
        return plots.plot_probes(volume, [spec], out, ndim=args.ndim,
                                 fields=_list(args.field), options=options)
    if cmd == 'dv-map':
        from mdao_su2.run_config import ConfigError, load_config
        try:
            cfg = load_config(args.config, overrides={}, check_files=False)
        except ConfigError as exc:
            raise UsageError(f'configuration: {exc}') from exc
        return plots.plot_dv_map(cfg, out, figures=figures, wall=args.wall, dvs=args.dvs,
                                 options=options)
    raise UsageError(f'unknown subcommand {cmd!r}')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    try:
        written = run(args, argv)
    except UsageError as exc:
        print(f'USAGE ERROR: {exc}', file=sys.stderr)
        return 2
    except Exception as exc:
        print(f'PLOT ERROR: {exc}', file=sys.stderr)
        return 1
    for name, path in written.items():
        print(f'  {name}: {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
