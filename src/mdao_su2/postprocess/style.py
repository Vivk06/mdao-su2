"""One palette, one style.  No hex colour literal belongs outside this file."""
from matplotlib.colors import LinearSegmentedColormap

__all__ = [
    'BASELINE', 'OPTIMIZED', 'BASELINE_MUTED', 'OPTIMIZED_ACCENT',
    'VALIDATED',
    'GRID', 'TEXT', 'LATTICE', 'EDGE',
    'FROZEN', 'FROZEN_DARK',
    'INACTIVE', 'INACTIVE_EDGE', 'LABEL',
    'diverging', 'DPI', 'FONT_SIZE',
    'footer', 'COMPARE_DEFAULT', 'COMPARE_CATEGORICAL', 'series_styles',
]

# --- semantic colours -------------------------------------------------------

BASELINE = '#2563eb'
OPTIMIZED = '#dc2626'
BASELINE_MUTED = '#9aa0a6'
OPTIMIZED_ACCENT = '#1f77b4'
VALIDATED = '#16a34a'

GRID = '#d1d5db'
TEXT = '#1f2937'
LATTICE = '#d0d4d8'
EDGE = '#4a4a4a'

FROZEN = '#943126'
FROZEN_DARK = '#5b1d16'
INACTIVE = '#c8ccce'
INACTIVE_EDGE = '#8a8f94'
LABEL = '#3c4043'

# --- comparison series ------------------------------------------------------

#: Default colours of a multi-run comparison, in entry order.
COMPARE_DEFAULT = ('#2563eb', '#dc2626', '#16a34a', '#f59e0b',
                   '#7c3aed', '#0891b2', '#db2777', '#4b5563')

#: Categorical set: the first colour is the reference or baseline grey.
COMPARE_CATEGORICAL = ('#52514e', '#2a78d6', '#eb6834', '#1baf7a', '#eda100',
                       '#e87ba4', '#008300', '#6250d6', '#e34948')

_PALETTES = {'default': COMPARE_DEFAULT, 'categorical': COMPARE_CATEGORICAL}

# --- defaults ----------------------------------------------------------------

DPI = 150
FONT_SIZE = 9

# --- colourmap ---------------------------------------------------------------

_DIV_ANCHORS = ['#1b4f72', '#5499c7', '#c8ccce', '#e59866', '#943126']


def diverging():
    """Two hues through a neutral grey midpoint (never a rainbow colourmap)."""
    return LinearSegmentedColormap.from_list('signed', _DIV_ANCHORS)


# --- provenance footer -------------------------------------------------------

def footer(fig, run_id=None, cfg_path=None, sha=None, cmd=None):
    """Stamp provenance on a figure so two runs never produce identical PNGs."""
    parts = []
    if run_id:
        parts.append(f'run={run_id}')
    if cfg_path:
        parts.append(cfg_path)
    if sha:
        parts.append(f'git:{sha[:8]}')
    if cmd:
        # N-file commands run to kilobytes; the full command is in manifest.json.
        parts.append(cmd if len(cmd) <= 100 else cmd[:99] + '… (full command: manifest.json)')
    if parts:
        # Below everything the figure draws (labels, legends), outside the axes area.
        box = fig.get_tightbbox(fig.canvas.get_renderer()).transformed(
            fig.transFigure.inverted())
        fig.text(0.01, min(box.y0, 0.0) - 0.04 / fig.get_figheight(), '  |  '.join(parts),
                 fontsize=6, color=TEXT, alpha=0.6,
                 va='top', ha='left', family='monospace', wrap=True)


# --- comparison colours and line styles -------------------------------------

def series_styles(count, palette='default', colors=None, styles=None):
    """``(colors, linestyles)`` for ``count`` comparison entries, in entry order.

    ``colors`` (hex or matplotlib names) replace the palette; ``styles`` are
    matplotlib linestyles and default to solid. Nothing cycles: fewer colours
    or styles than entries, or an unknown name, raises ``ValueError``.
    """
    from matplotlib.colors import is_color_like
    from matplotlib.lines import Line2D
    if palette not in _PALETTES:
        raise ValueError(f'palette {palette!r} must be one of {sorted(_PALETTES)}')
    source = list(colors) if colors else list(_PALETTES[palette])
    where = '--colors' if colors else f'the {palette!r} palette'
    if count > len(source):
        raise ValueError(f'{count} entries but {where} has {len(source)} colours; '
                         f'give --colors with at least {count}')
    bad = [c for c in source[:count] if not is_color_like(c)]
    if bad:
        raise ValueError(f'not colours: {bad}')
    if styles:
        if count > len(styles):
            raise ValueError(f'{count} entries but --styles has {len(styles)} entries')
        for item in styles[:count]:
            try:
                Line2D([], []).set_linestyle(item)
            except ValueError:
                raise ValueError(f'not a matplotlib linestyle: {item!r}') from None
        lines = list(styles[:count])
    else:
        lines = ['-'] * count
    return source[:count], lines
