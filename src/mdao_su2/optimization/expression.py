"""Objective expression compiler: TOML text -> ``ExecComp`` source.

The grammar is one node table (``_NODES``); a new construct is one more entry.
Every rejection raises :class:`ExpressionError` with the column of the node.
``compile_linear`` builds the same record for a ``[[constraints.linear]]``.
"""

import ast
import cmath
import math
from dataclasses import dataclass, field

__all__ = ['ExpressionError', 'Compiled', 'compile_expression', 'compile_linear',
           'LEGACY_OUTPUTS', 'C1_FUNCTIONS', 'DOMAIN_FUNCTIONS']

#: Outputs every solver component declares; other catalog outputs need the
#: point to request the response so the router builds ResponseSU2AeroComp.
LEGACY_OUTPUTS = ('C_D', 'C_L', 'C_M')
#: Functions allowed only in C¹ position (base of an integer power >= 2).
C1_FUNCTIONS = ('maximum', 'minimum')
#: Functions defined only for a positive argument; each argument is checked
#: before the objective is evaluated.
DOMAIN_FUNCTIONS = ('log',)
_RESERVED = ('wsum',) + DOMAIN_FUNCTIONS + C1_FUNCTIONS
_BINOPS = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/', ast.Pow: '**'}


class ExpressionError(ValueError):
    """An expression outside the grammar; ``col`` is the 0-based column."""

    def __init__(self, message, col):
        self.col = col
        super().__init__(f'{message} (column {col})')


@dataclass(frozen=True)
class Compiled:
    text: str
    execcomp_src: str
    inputs: tuple
    connections: tuple                 # ((source path, 'obj_comb.<input>'), ...)
    base_functionals: dict             # point id -> tuple of outputs
    rendered: str                      # the same formula with point ids
    responses: tuple                   # catalog names used
    parameters: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)   # input -> (point id, output)
    #: Each ``log`` argument as (ExecComp text, rendered text with point ids).
    domain: tuple = ()

    def domain_errors(self, values):
        """Messages for every ``log`` argument that is not positive at ``values``."""
        errors = []
        for src, shown in self.domain:
            value = _run(f'obj = {src}', values)
            if not value.real > 0.0:
                errors.append(f'log argument {shown} = {value.real:.6g} is not positive')
        return errors

    def evaluate(self, values):
        """The ``ExecComp`` output for ``{input name: value}``; a ``log`` of a
        non-positive argument raises ``ValueError`` naming it."""
        errors = self.domain_errors(values)
        if errors:
            raise ValueError('; '.join(errors))
        return _run(self.execcomp_src, values)

    def partials(self, values, step=1e-30):
        """``{input: d output / d input}`` at ``values``, by complex step."""
        out = {}
        for name in self.inputs:
            shifted = dict(values)
            shifted[name] = complex(values[name], step)
            out[name] = _run(self.execcomp_src, shifted).imag / step
        return out


class _Compiler:
    def __init__(self, text, parameters, catalog, points, prefix):
        self.text = text
        self.parameters = {str(k): float(v) for k, v in (parameters or {}).items()}
        self.catalog = catalog
        from ..run_config.points import weight_total
        self.weighted = [(i, p) for i, p in enumerate(points) if p.weight > 0.0]
        self.total = weight_total(p for _, p in self.weighted)
        self.prefix = prefix
        self.inputs = {}               # input name -> (point id, output)
        self.responses = {}            # catalog name -> output
        self.domain = []               # (ExecComp text, rendered text) per log argument
        for name in self.parameters:
            if name in _RESERVED:
                raise ExpressionError(
                    f'parameter {name!r} shadows the function {name}()', 0)
            if self._response(name) is not None:
                raise ExpressionError(
                    f'parameter {name!r} shadows the response of the same name', 0)
            if not math.isfinite(self.parameters[name]):
                raise ExpressionError(f'parameter {name!r} is not finite', 0)

    # ------------------------------------------------------------ helpers
    def _response(self, name):
        try:
            return self.catalog(name)
        except ValueError:
            return None

    @staticmethod
    def _col(node):
        return getattr(node, 'col_offset', 0)

    def _const(self, node):
        """Value of a literal or parameter node, else None."""
        if isinstance(node, ast.Constant) and _is_number(node.value):
            return node.value
        if isinstance(node, ast.Name) and node.id in self.parameters:
            return self.parameters[node.id]
        return None

    # ------------------------------------------------------------- render
    def render(self, node, ctx):
        """Return (execcomp text, rendered text). ``ctx`` keys: point, c1."""
        handler = _NODES.get(type(node))
        if handler is None:
            raise ExpressionError(_rejection(node), self._col(node))
        return handler(self, node, ctx)

    def _name(self, node, ctx):
        if node.id in self.parameters:
            value = self.parameters[node.id]
            text = repr(value) if value >= 0 else f'({value!r})'   # -0.5**2 != 0.25
            return text, text
        if node.id in _RESERVED:
            raise ExpressionError(f'{node.id} must be called', self._col(node))
        record = self._response(node.id)
        if record is None:
            raise ExpressionError(
                f'unknown name {node.id!r}: not a response or an '
                f'[objective.parameters] entry', self._col(node))
        if ctx.get('point') is None:
            raise ExpressionError(
                f'response {node.id!r} outside wsum(...): responses are '
                f'aggregated over points only through wsum', self._col(node))
        i, point = ctx['point']
        output = record.output
        if output not in LEGACY_OUTPUTS:
            # An unknown name in point.responses is the point validator's error.
            requested = {record.output for record in map(self._response, point.responses)
                         if record is not None}
            if output not in requested:
                raise ExpressionError(
                    f'response {node.id!r} ({output}) is not in point '
                    f'{point.id!r} responses; add it there so the solver '
                    f'component provides it', self._col(node))
        self.responses.setdefault(node.id, output)
        name = f'{output}_{i}'
        self.inputs[name] = (point.id, output)
        return name, f'{node.id}[{point.id}]'

    def _constant(self, node, ctx):
        if not _is_number(node.value):
            raise ExpressionError(f'literal {node.value!r} is not a number',
                                  self._col(node))
        return repr(node.value), repr(node.value)

    def _unary(self, node, ctx):
        if not isinstance(node.op, ast.USub):
            raise ExpressionError('only unary minus is allowed', self._col(node))
        a, r = self.render(node.operand, ctx)
        return f'(-{a})', f'(-{r})'

    def _binop(self, node, ctx):
        op = type(node.op)
        if op not in _BINOPS:
            raise ExpressionError(f'operator {type(node.op).__name__} is not '
                                  f'allowed', self._col(node))
        if op is ast.Pow:
            return self._pow(node, ctx)
        inner = dict(ctx)
        if op is ast.Div:
            value = self._const(node.right)
            if value is None:
                raise ExpressionError(
                    'the divisor must be a literal or a parameter; a response '
                    'divisor is not supported', self._col(node.right))
            if value == 0:
                raise ExpressionError('division by zero', self._col(node.right))
        if op in (ast.Add, ast.Sub):
            inner['c1'] = False        # (max(x, 0) + y)**2 is not C¹
        elif op is ast.Mult and ctx.get('c1'):
            if self._const(node.left) is None and self._const(node.right) is None:
                inner['c1'] = False
        left = self.render(node.left, inner)
        right = self.render(node.right, inner if op is not ast.Div else
                            dict(inner, c1=False))
        sym = _BINOPS[op]
        return (f'{_wrap(node.left, left[0])} {sym} {_wrap(node.right, right[0])}',
                f'{_wrap(node.left, left[1])} {sym} {_wrap(node.right, right[1])}')

    def _pow(self, node, ctx):
        exp = node.right
        if not (isinstance(exp, ast.Constant) and _is_number(exp.value)):
            raise ExpressionError('the exponent must be an integer literal',
                                  self._col(exp))
        if isinstance(exp.value, float) and not exp.value.is_integer():
            raise ExpressionError(
                f'fractional exponent {exp.value!r}; fractional powers are not '
                f'supported', self._col(exp))
        if exp.value < 0:
            raise ExpressionError('a negative exponent divides by its base; '
                                  'response divisors are not supported', self._col(exp))
        p = int(exp.value)
        base = self.render(node.left, dict(ctx, c1=p >= 2))
        e = repr(exp.value)
        return (f'{_wrap(node.left, base[0])}**{e}',
                f'{_wrap(node.left, base[1])}**{e}')

    def _call(self, node, ctx):
        if not isinstance(node.func, ast.Name) or node.func.id not in _RESERVED:
            name = getattr(node.func, 'id', type(node.func).__name__)
            raise ExpressionError(f'function {name!r} is not allowed; the grammar allows '
                                  f'wsum, log, maximum, minimum', self._col(node))
        if node.keywords:
            raise ExpressionError('keyword arguments are not allowed',
                                  self._col(node.keywords[0].value))
        if node.func.id == 'wsum':
            return self._wsum(node, ctx)
        if node.func.id == 'log':
            return self._log(node, ctx)
        return self._c1(node, ctx)

    def _log(self, node, ctx):
        if len(node.args) != 1:
            raise ExpressionError('log takes one argument', self._col(node))
        a, r = self.render(node.args[0], dict(ctx, c1=False))
        self.domain.append((a, r))
        return f'log({a})', f'log({r})'

    def _c1(self, node, ctx):
        fn = node.func.id
        if len(node.args) != 2:
            raise ExpressionError(f'{fn} takes two arguments', self._col(node))
        if not ctx.get('c1'):
            raise ExpressionError(
                f'{fn}(...) must be the base of an integer power >= 2, scaled '
                f'only by literals or parameters, so the objective is C¹',
                self._col(node))
        zero = [isinstance(a, ast.Constant) and _is_number(a.value) and a.value == 0
                for a in node.args]
        if not any(zero):
            raise ExpressionError(
                f'{fn}(...) needs the literal 0 as one argument; otherwise the '
                f'squared kink is not C¹', self._col(node))
        inner = dict(ctx, c1=False)
        args = [self.render(a, inner) for a in node.args]
        return (f'{fn}({args[0][0]}, {args[1][0]})',
                f'{fn}({args[0][1]}, {args[1][1]})')

    def _wsum(self, node, ctx):
        if ctx.get('point') is not None:
            raise ExpressionError('wsum may not nest', self._col(node))
        if len(node.args) != 1:
            raise ExpressionError('wsum takes one argument', self._col(node))
        if not self.weighted:
            raise ExpressionError('wsum needs at least one point with weight > 0',
                                  self._col(node))
        arg = node.args[0]
        src, shown = [], []
        for i, point in self.weighted:
            a, r = self.render(arg, dict(ctx, point=(i, point), c1=False))
            a, r = _wrap(arg, a), _wrap(arg, r)
            w = float(point.weight)
            src.append(a if w == 1.0 else f'{w}*{a}')
            shown.append(r if w == 1.0 else f'{w}*{r}')
        total = self.total
        return (f'({" + ".join(src)}) / {total}',
                f'({" + ".join(shown)}) / {total}')

    def _attribute(self, node, ctx):
        raise ExpressionError(
            'point-qualified responses (response.point) are not supported; use wsum',
            self._col(node))


def _pick(a, b, larger):
    """Complex-safe max/min on the real part (ExecComp does the same)."""
    return a if (a.real >= b.real) == larger else b


def _run(src, values):
    scope = dict(values,
                 maximum=lambda a, b: _pick(a, b, True),
                 minimum=lambda a, b: _pick(a, b, False),
                 log=lambda a: cmath.log(a) if isinstance(a, complex) else math.log(a))
    exec(src, {'__builtins__': {}}, scope)
    return scope[src.partition('=')[0].strip()]


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _wrap(node, text):
    """Parenthesize compound operands so the rendered text keeps the AST tree."""
    if isinstance(node, (ast.BinOp, ast.UnaryOp)) or (
            isinstance(node, ast.Call) and getattr(node.func, 'id', '') == 'wsum'):
        return text if text.startswith('(') and _balanced(text) else f'({text})'
    return text


def _balanced(text):
    """True when the leading '(' closes at the last character."""
    depth = 0
    for k, ch in enumerate(text):
        depth += ch == '('
        depth -= ch == ')'
        if depth == 0:
            return k == len(text) - 1
    return False


def _rejection(node):
    kind = type(node).__name__
    reasons = {
        'Compare': 'comparisons are not allowed',
        'BoolOp': 'boolean operators are not allowed',
        'Subscript': 'subscripts are not allowed',
        'Lambda': 'lambdas are not allowed',
        'IfExp': 'conditional expressions are not allowed',
    }
    if kind.endswith('Comp'):
        return 'comprehensions are not allowed'
    return reasons.get(kind, f'{kind} is not allowed in an objective expression')


#: The grammar: one entry per accepted AST node type.
_NODES = {
    ast.Name: _Compiler._name,
    ast.Constant: _Compiler._constant,
    ast.UnaryOp: _Compiler._unary,
    ast.BinOp: _Compiler._binop,
    ast.Call: _Compiler._call,
    ast.Attribute: _Compiler._attribute,
}


def compile_expression(text, parameters, catalog, points, prefix=''):
    """Compile ``text`` over ``points`` (objects with id, weight, responses).

    ``catalog`` maps a response name to a record with ``output`` and raises
    ``ValueError`` for an unknown name (``response_catalog.lookup``).
    ``prefix`` is the scenario path prefix used in the connections.
    """
    text = str(text).strip()
    try:
        tree = ast.parse(text, mode='eval')
    except SyntaxError as exc:
        raise ExpressionError(f'syntax error: {exc.msg}',
                              max((exc.offset or 1) - 1, 0)) from None
    compiler = _Compiler(text, parameters, catalog, points, prefix)
    src, shown = compiler.render(tree.body, {'point': None, 'c1': False})
    if not compiler.inputs:
        raise ExpressionError('the objective uses no response; wrap one in '
                              'wsum(...)', 0)
    inputs = tuple(compiler.inputs)
    base = {}
    for pid, output in compiler.inputs.values():
        base.setdefault(pid, [])
        if output not in base[pid]:
            base[pid].append(output)
    return Compiled(
        text=text,
        execcomp_src=f'obj = {src}',
        inputs=inputs,
        connections=tuple((f'{prefix}{pid}.{out}', f'obj_comb.{name}')
                          for name, (pid, out) in compiler.inputs.items()),
        base_functionals={pid: tuple(v) for pid, v in base.items()},
        rendered=f'obj = {shown}',
        responses=tuple(compiler.responses),
        parameters=dict(compiler.parameters),
        sources=dict(compiler.inputs),
        domain=tuple(compiler.domain))


def compile_linear(constraint, catalog, points, prefix=''):
    """``value = sum(coeff * response(point)) + offset`` for one
    ``[[constraints.linear]]`` as a :class:`Compiled`.

    Inputs are named ``<output>_<point index>``, as in the objective; ``prefix``
    is the scenario path prefix of the connections. The load-time checks
    (``run_config/checks/points.py``) have already rejected unknown points and
    responses, so a lookup failure here is a caller error.
    """
    index = {p.id: i for i, p in enumerate(points)}
    sources, src, shown, base, responses = {}, [], [], {}, {}
    for term in constraint.terms:
        output = catalog(term.response).output
        name = f'{output}_{index[term.point]}'
        sources[name] = (term.point, output)
        coeff = repr(float(term.coeff))
        coeff = coeff if term.coeff >= 0 else f'({coeff})'
        src.append(f'{coeff}*{name}')
        shown.append(f'{coeff}*{term.response}[{term.point}]')
        base.setdefault(term.point, [])
        if output not in base[term.point]:
            base[term.point].append(output)
        responses.setdefault(term.response, output)
    offset = repr(float(constraint.offset))
    offset = offset if constraint.offset >= 0 else f'({offset})'
    component = constraint.component
    return Compiled(
        text=constraint.name,
        execcomp_src=f'value = {" + ".join(src)} + {offset}',
        inputs=tuple(sources),
        connections=tuple((f'{prefix}{pid}.{out}', f'{component}.{name}')
                          for name, (pid, out) in sources.items()),
        base_functionals={pid: tuple(v) for pid, v in base.items()},
        rendered=f'value = {" + ".join(shown)} + {offset}',
        responses=tuple(responses),
        sources=sources)
