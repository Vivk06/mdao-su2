"""Metadata for every SU2 v8.5 objective, keyed from its generated enum."""

from dataclasses import dataclass

from . import su2_options, su2_options_v85


@dataclass(frozen=True)
class Response:
    name: str
    output: str
    su2_objective: str
    history_field: str
    requires: str
    alpha_rule: str
    dimension: str
    scope: str
    statuses: dict
    evidence: dict


MAPPED = {
    'DRAG': ('drag', 'C_D', 'DRAG', 'monitor', 'wind_drag'),
    'LIFT': ('lift', 'C_L', 'LIFT', 'monitor', 'wind_lift'),
    'SIDEFORCE': ('sideforce', 'C_Y', 'SIDEFORCE', 'monitor_3d', 'body'),
    'EFFICIENCY': ('efficiency', 'C_EFF', 'EFFICIENCY', 'monitor', 'wind_efficiency'),
    'MOMENT_X': ('moment_x', 'C_MX', 'MOMENT_X', 'monitor_3d', 'moment'),
    'MOMENT_Y': ('moment_y', 'C_MY', 'MOMENT_Y', 'monitor_3d', 'moment'),
    'MOMENT_Z': ('moment_z', 'C_MZ', 'MOMENT_Z', 'monitor', 'moment'),
    'FORCE_X': ('force_x', 'C_FX', 'FORCE_X', 'monitor', 'body'),
    'FORCE_Y': ('force_y', 'C_FY', 'FORCE_Y', 'monitor_3d', 'body'),
    'FORCE_Z': ('force_z', 'C_FZ', 'FORCE_Z', 'monitor', 'body'),
    # Separation sensor of the monitored viscous walls; SU2 registers its
    # history field only for viscous solvers (CFlowCompOutput.cpp:197-199).
    'BUFFET': ('buffet', 'C_BUFFET', 'BUFFET', 'monitor_viscous', 'freestream'),
}
_STATUS = {row.subject: row for row in su2_options.STATUS}


def make_response(objective):
    item = MAPPED.get(objective)
    if item:
        name, output, field, requires, alpha = item
    else:
        name = objective.lower()
        output = 'R_' + objective
        field = objective
        requires = 'unmapped'
        alpha = 'none'
    statuses, evidence = {}, {}
    for profile in ('euler', 'rans_sa', 'rans:sa[none]+none[none]',
                    'rans:sa[none]+lm[none]', 'rans:sst[none]+lm[none]',
                    'rans:sst[v1994m]+lm[none]'):
        row = _STATUS.get(f'response:{objective}@{profile}')
        alias = {'rans:sa[none]+none[none]': 'rans_sa',
                 'rans:sa[none]+lm[none]': 'rans_sa_lm',
                 'rans:sst[none]+lm[none]': 'rans_sst_lm',
                 'rans:sst[v1994m]+lm[none]': 'rans_sst_lm'}.get(profile)
        alias_row = _STATUS.get(f'response:{objective}@{alias}') if alias else None
        profile_row = _STATUS.get(f'profile:{alias}') if alias else None
        fallback = profile_row if profile_row and profile_row.adjoint == 'unsupported' else None
        selected = row or alias_row or fallback
        statuses[profile] = selected.adjoint if selected else 'assumed'
        evidence[profile] = selected.evidence if selected else ()
    return Response(name, output, objective, field, requires, alpha,
                    'coefficient', 'monitored', statuses, evidence)


CATALOG = tuple(make_response(item) for item in su2_options_v85.ENUMS['Objective_Map'])
_BY_NAME = {item.name: item for item in CATALOG}
_MOMENT_Z = _BY_NAME['moment_z']
_LEGACY_MOMENT = Response('moment', 'C_M', 'MOMENT_Z', 'MOMENT_Z',
                          'monitor', 'moment', 'coefficient', 'monitored',
                          _MOMENT_Z.statuses, _MOMENT_Z.evidence)


def lookup(name):
    key = str(name).lower()
    if key == 'moment':
        return _LEGACY_MOMENT
    try:
        return _BY_NAME[key]
    except KeyError:
        raise ValueError(f'unknown response {name!r}; known: {sorted(_BY_NAME)}') from None


def for_profile(name, profile):
    return status(name, profile)[0]


def status(name, profile):
    """Return (status, evidence) for a response/profile pair."""
    item = lookup(name)
    state = item.statuses.get(profile, 'assumed')
    evidence = item.evidence.get(profile, ())
    if state == 'assumed' and profile.startswith('rans:') and '+lm[' in profile:
        family = 'rans_sst_lm' if profile.startswith('rans:sst[') else 'rans_sa_lm'
        record = _STATUS[f'profile:{family}']
        return record.adjoint, record.evidence
    return state, evidence


def supports(item, cfg):
    if item.requires == 'unmapped':
        return False
    if item.requires.endswith('_3d') and cfg.case.ndim != 3:
        return False
    if item.requires == 'monitor_viscous':
        return not cfg.case.physics.inviscid and bool(cfg.case.markers_with('monitor'))
    if item.requires == 'monitor' or item.requires.endswith('_3d'):
        return bool(cfg.case.markers_with('monitor'))
    return True


__all__ = ['Response', 'MAPPED', 'make_response', 'CATALOG', 'lookup', 'for_profile', 'status', 'supports']
