"""L-BFGS-B — limited-memory BFGS for bound-constrained optimization.

# C++ parity: ql/math/optimization/lbfgsb.{hpp,cpp} (v1.43) — new in that release.

Faithful port of Byrd, Lu, Nocedal and Zhu, "A Limited Memory Algorithm
for Bound Constrained Optimization", SIAM J. Sci. Comput. 16(5):1190-1208,
1995, as implemented in C++ QuantLib v1.43.

Structure, mirroring the C++ translation unit:

- ``_build_compact_rep`` — the compact limited-memory representation
  ``B = theta I - W M W^T`` (eq. 3.5) built from the stored correction pairs.
- ``_generalized_cauchy_point`` — Algorithm CP: walk the breakpoints of the
  projected steepest-descent path, pinning variables as they reach a bound.
- ``_subspace_minimization`` — the direct primal method of section 5.1 over
  the free variables, truncated back into the box.
- ``_line_search_wolfe`` — strong Wolfe (Nocedal & Wright 3.5/3.6) capped at
  the largest feasible step.

Details that a port silently gets wrong, reproduced deliberately and pinned
by the ``v143/lbfgsb`` probe:

- ``Problem.set_gradient_norm_value`` stores **pgInf squared** — the square of
  the infinity norm of the *projected* gradient, which is the KKT residual of
  a box-constrained problem, not ``|g|^2``. ``EndCriteria.check_zero_gradient_norm``
  however receives pgInf itself, unsquared.
- ``factr = f_tol / QL_EPSILON`` and the stop test multiplies by ``QL_EPSILON``
  again. The divide-then-multiply round trip is reproduced literally;
  collapsing it to ``f_tol * denom`` moves the last bits.
- Explicit constructor bounds **override** the problem's constraint; they are
  not intersected with it.
- A bound is "absent" at ``u >= 0.5 * QL_MAX_REAL`` / ``l <= -0.5 * QL_MAX_REAL``.
  The 0.5 guards against overflow in ``x - bound``; testing ``== QL_MAX_REAL``
  is not the same predicate.
- The start point is clipped into the box *before* the first evaluation.
- The objective is reached only through ``Problem.value_and_gradient``, never
  through ``value`` / ``gradient`` separately. A value-only cost function's
  central-difference gradient therefore runs inside the cost function and does
  not move the problem's evaluation counters.

Python-specific translations:

- C++ ``Array`` / ``Matrix`` are numpy arrays (see ``pquantlib.math.array``).
- C++ ``inverse(Matrix)`` (uBLAS LU factorization + substitution against the
  identity) becomes ``_inverse`` below, which asks LAPACK for the same thing.
- C++'s ``EndCriteria::Type&`` out-parameters become ``Type | None`` returns.
- C++'s out-parameter triples become small dataclasses (``_CauchyPoint``,
  ``_LineSearchResult``).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.constants import QL_EPSILON, QL_MAX_REAL
from pquantlib.math.matrix import Matrix
from pquantlib.math.optimization.end_criteria import Type
from pquantlib.math.optimization.optimization_method import OptimizationMethod

if TYPE_CHECKING:
    from pquantlib.math.optimization.end_criteria import EndCriteria
    from pquantlib.math.optimization.problem import Problem

# C++ parity: lbfgsb.cpp:31 — ``const Real INF = QL_MAX_REAL;``
_INF: Final[float] = QL_MAX_REAL


def _no_upper(u: float) -> bool:
    """True when ``u`` is the "no upper bound" sentinel.

    # C++ parity: lbfgsb.cpp:36-38 — ``u >= 0.5 * INF``.

    The 0.5 factor is not cosmetic: it keeps ``x - u`` from overflowing for a
    coordinate sitting anywhere in the representable range.
    """
    return u >= 0.5 * _INF


def _no_lower(low: float) -> bool:
    """True when ``low`` is the "no lower bound" sentinel.

    # C++ parity: lbfgsb.cpp:39-41 — ``l <= -0.5 * INF``.
    """
    return low <= -0.5 * _INF


def _inverse(m: Matrix) -> Matrix:
    """Matrix inverse by LU factorization with partial pivoting.

    # C++ parity: ql/math/matrix.cpp:44-79 — ``QuantLib::inverse``, which
    # runs uBLAS ``lu_factorize`` and then ``lu_substitute`` against the
    # identity. ``numpy.linalg.solve`` against the identity is LAPACK's
    # ``gesv``: the same factor-then-substitute sequence.

    Raises ``numpy.linalg.LinAlgError`` on an exactly singular matrix, which
    is where C++ raises ``"singular matrix given"``. Both are caught by the
    correction-pair eviction loop in ``LBFGSB.minimize``.
    """
    n = m.shape[0]
    return np.asarray(np.linalg.solve(m, np.eye(n, dtype=np.float64)), dtype=np.float64)


@dataclass(slots=True)
class _CompactRep:
    """Compact limited-memory representation of the BFGS Hessian model.

    # C++ parity: lbfgsb.cpp:47-53 — ``struct CompactRep``.

    ``B = theta I - W M W^T`` (Byrd, Lu, Nocedal & Zhu 1995, eq. 3.5), where
    ``w`` is ``n x 2col`` holding ``[ Y | theta S ]`` and ``m_inv`` is the
    inverse of the ``2col x 2col`` middle matrix.
    """

    w: Matrix = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float64))
    m_inv: Matrix = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float64))
    theta: float = 1.0
    col: int = 0


def _build_compact_rep(s_pairs: list[Array], y_pairs: list[Array], theta: float) -> _CompactRep:
    """Assemble the compact representation from the stored correction pairs.

    # C++ parity: lbfgsb.cpp:55-105 — ``buildCompactRep``.
    """
    rep = _CompactRep(theta=theta, col=len(s_pairs))
    col = rep.col
    if col == 0:
        return rep

    n = s_pairs[0].size
    w = np.zeros((n, 2 * col), dtype=np.float64)
    for j in range(col):
        w[:, j] = y_pairs[j]
        w[:, col + j] = theta * s_pairs[j]

    # small Gram matrices S^T S and S^T Y
    sts = np.zeros((col, col), dtype=np.float64)
    sty = np.zeros((col, col), dtype=np.float64)
    for i in range(col):
        for j in range(col):
            sts[i, j] = float(np.dot(s_pairs[i], s_pairs[j]))
            sty[i, j] = float(np.dot(s_pairs[i], y_pairs[j]))

    # middle matrix [ -D  L^T ; L  theta S^T S ] with D = diag(s_i.y_i) and
    # L the strictly lower part of S^T Y.
    mid = np.zeros((2 * col, 2 * col), dtype=np.float64)
    for i in range(col):
        mid[i, i] = -sty[i, i]  # -D
        for j in range(col):
            if i > j:
                mid[col + i, j] = sty[i, j]  # L
            if j > i:
                mid[i, col + j] = sty[j, i]  # L^T
            mid[col + i, col + j] = theta * sts[i, j]

    rep.w = w
    rep.m_inv = _inverse(mid)
    return rep


@dataclass(slots=True)
class _CauchyPoint:
    """Outcome of the generalized-Cauchy-point walk.

    ``xcp`` is the Cauchy point, ``c`` the accumulated ``W^T (xcp - x)`` the
    subspace step needs, and ``is_free`` marks the variables not pinned at a
    bound. C++ returns these through three out-parameters.
    """

    xcp: Array
    c: Array
    is_free: npt.NDArray[np.bool_]


def _generalized_cauchy_point(  # noqa: PLR0915 — one-to-one with the C++ function body
    x: Array, g: Array, lo: Array, hi: Array, rep: _CompactRep
) -> _CauchyPoint:
    """Generalized Cauchy point — Byrd et al. 1995, Algorithm CP.

    # C++ parity: lbfgsb.cpp:115-247 — ``generalizedCauchyPoint``.

    Walks the breakpoints of the projected steepest-descent path
    ``x(t) = P(x - t g, l, u)`` and stops at the first local minimizer of the
    quadratic model along it.
    """
    n = x.size
    m2 = 2 * rep.col

    xcp = x.astype(np.float64, copy=True)
    is_free = np.ones(n, dtype=np.bool_)

    t = np.empty(n, dtype=np.float64)
    d = np.zeros(n, dtype=np.float64)
    brk: list[int] = []  # indices with a strictly positive breakpoint
    for i in range(n):
        gi = float(g[i])
        if gi < 0.0:
            ti = _INF if _no_upper(float(hi[i])) else (float(x[i]) - float(hi[i])) / gi
        elif gi > 0.0:
            ti = _INF if _no_lower(float(lo[i])) else (float(x[i]) - float(lo[i])) / gi
        else:
            ti = _INF
        t[i] = ti

        if ti <= 0.0:
            # already sitting on the bound the gradient pushes into
            d[i] = 0.0
            is_free[i] = False
        else:
            d[i] = -gi
            brk.append(i)

    c = np.zeros(m2, dtype=np.float64)
    if not brk:  # every variable is pinned
        return _CauchyPoint(xcp=xcp, c=c, is_free=is_free)

    brk.sort(key=lambda i: float(t[i]))

    p = d @ rep.w if rep.col > 0 else np.zeros(0, dtype=np.float64)

    fp = -float(np.dot(d, d))  # first derivative of the model, m'(0)
    fpp = -rep.theta * fp  # second derivative, theta d^T d - ...
    if rep.col > 0:
        fpp -= float(np.dot(p, rep.m_inv @ p))
    fpp_floor = QL_EPSILON * (fpp if fpp > 0.0 else 1.0)

    dt_min = -fp / fpp if fpp > 0.0 else 0.0

    t_old = 0.0
    ptr = 0
    b = brk[ptr]
    tb = float(t[b])
    dt = tb
    exhausted = False

    while dt_min >= dt and tb < _INF:
        # pin variable b at the bound it has reached
        db = float(d[b])
        if db > 0.0:
            xcpb = float(hi[b])
        elif db < 0.0:
            xcpb = float(lo[b])
        else:
            xcpb = float(x[b])
        xcp[b] = xcpb
        is_free[b] = False
        zb = xcpb - float(x[b])
        gb = float(g[b])

        if rep.col > 0:
            c = c + dt * p

            wb = rep.w[b]
            mc = rep.m_inv @ c
            mp = rep.m_inv @ p
            mwb = rep.m_inv @ wb

            fp += dt * fpp + gb * gb + rep.theta * gb * zb - gb * float(np.dot(wb, mc))
            fpp += (
                -rep.theta * gb * gb
                - 2.0 * gb * float(np.dot(wb, mp))
                - gb * gb * float(np.dot(wb, mwb))
            )
            p = p + gb * wb
        else:
            fp += dt * fpp + gb * gb + rep.theta * gb * zb
            fpp += -rep.theta * gb * gb

        fpp = max(fpp, fpp_floor)

        d[b] = 0.0
        dt_min = -fp / fpp if fpp > 0.0 else 0.0
        t_old = tb

        ptr += 1
        if ptr >= len(brk):
            exhausted = True
            break

        b = brk[ptr]
        tb = float(t[b])
        dt = tb - t_old

    # advance the still-free variables along the final segment
    if exhausted:
        dt_min = 0.0
    dt_min = max(dt_min, 0.0)
    t_old += dt_min

    xcp = np.where(is_free, x + t_old * d, xcp)

    if rep.col > 0:
        c = c + dt_min * p
    return _CauchyPoint(xcp=xcp, c=c, is_free=is_free)


def _subspace_minimization(
    x: Array,
    g: Array,
    xcp: Array,
    c: Array,
    lo: Array,
    hi: Array,
    rep: _CompactRep,
    is_free: npt.NDArray[np.bool_],
) -> Array:
    """Direct primal subspace minimization — Byrd et al. 1995, section 5.1.

    # C++ parity: lbfgsb.cpp:249-359 — ``subspaceMinimization``.

    Minimizes the quadratic model over the free variables with the active ones
    held at their Cauchy-point bound, then truncates the step back into the box.
    """
    n = x.size
    free_idx = [i for i in range(n) if bool(is_free[i])]
    nf = len(free_idx)

    xbar = xcp.astype(np.float64, copy=True)
    if nf == 0:
        return xbar

    theta = rep.theta
    m2 = 2 * rep.col

    # reduced gradient of the model at the Cauchy point:
    #   r = [ g + theta (xcp - x) - W M c ] restricted to the free variables
    wmc = rep.w @ (rep.m_inv @ c) if rep.col > 0 else np.zeros(n, dtype=np.float64)

    r = np.empty(nf, dtype=np.float64)
    for k, i in enumerate(free_idx):
        r[k] = float(g[i]) + theta * (float(xcp[i]) - float(x[i])) - float(wmc[i])

    dhat = np.empty(nf, dtype=np.float64)
    if rep.col == 0:
        for k in range(nf):
            dhat[k] = -float(r[k]) / theta
    else:
        # v = M (W_free^T r)
        wtr = np.zeros(m2, dtype=np.float64)
        for k, i in enumerate(free_idx):
            for j in range(m2):
                wtr[j] += float(rep.w[i, j]) * float(r[k])

        v = rep.m_inv @ wtr

        # N = I - (1/theta) M (W_free^T W_free)
        wftwf = np.zeros((m2, m2), dtype=np.float64)
        for i in free_idx:
            for a in range(m2):
                for bb in range(m2):
                    wftwf[a, bb] += float(rep.w[i, a]) * float(rep.w[i, bb])

        n_mat = rep.m_inv @ wftwf
        n_mat *= -1.0 / theta
        for a in range(m2):
            n_mat[a, a] += 1.0
        v = _inverse(n_mat) @ v

        # dhat = -(1/theta) r - (1/theta^2) W_free v
        for k, i in enumerate(free_idx):
            wfv = 0.0
            for j in range(m2):
                wfv += float(rep.w[i, j]) * float(v[j])
            dhat[k] = -float(r[k]) / theta - wfv / (theta * theta)

    # truncate so that every free variable stays inside the box
    alpha_star = 1.0
    for k, i in enumerate(free_idx):
        dk = float(dhat[k])
        if dk > 0.0 and not _no_upper(float(hi[i])):
            alpha_star = min(alpha_star, (float(hi[i]) - float(xcp[i])) / dk)
        elif dk < 0.0 and not _no_lower(float(lo[i])):
            alpha_star = min(alpha_star, (float(lo[i]) - float(xcp[i])) / dk)
    alpha_star = max(alpha_star, 0.0)

    for k, i in enumerate(free_idx):
        xbar[i] = float(xcp[i]) + alpha_star * float(dhat[k])
    return xbar


def _max_feasible_step(x: Array, d: Array, lo: Array, hi: Array) -> float:
    """Largest step length along ``d`` from ``x`` that stays inside the box.

    # C++ parity: lbfgsb.cpp:361-373 — ``maxFeasibleStep``.
    """
    stp = _INF
    for i in range(x.size):
        di = float(d[i])
        if di > 0.0 and not _no_upper(float(hi[i])):
            stp = min(stp, (float(hi[i]) - float(x[i])) / di)
        elif di < 0.0 and not _no_lower(float(lo[i])):
            stp = min(stp, (float(lo[i]) - float(x[i])) / di)
    return stp


@dataclass(slots=True)
class _LineSearchResult:
    """The line search's four out-parameters."""

    alpha: float
    xt: Array
    ft: float
    gt: Array


def _line_search_wolfe(  # noqa: PLR0915 — one-to-one with the C++ function body
    problem: Problem,
    x: Array,
    d: Array,
    f0: float,
    g0: Array,
    stp_max: float,
) -> _LineSearchResult | None:
    """Strong-Wolfe line search capped at the largest feasible step.

    # C++ parity: lbfgsb.cpp:375-484 — ``lineSearchWolfe``.

    ``c1 = 1e-4``, ``c2 = 0.9``, at most 30 iterations per phase, initial step
    ``min(1, stp_max)``, doubling expansion then bisection. When strong Wolfe
    is never reached the best *sufficient decrease* seen is accepted; only a
    search that never improved on ``f0`` at all fails (returns ``None``).
    """
    c1 = 1e-4
    c2 = 0.9
    max_iter = 30

    dphi0 = float(np.dot(g0, d))
    if dphi0 >= 0.0:
        return None  # not a descent direction

    xt: Array = x.astype(np.float64, copy=True)
    gt: Array = np.zeros(x.size, dtype=np.float64)
    ft: float = f0

    have_best = False
    best_alpha = 0.0
    best_f = f0
    best_x: Array = xt
    best_g: Array = gt

    def evaluate(a: float) -> float:
        """phi'(a), evaluating phi(a) into ``ft`` / ``xt`` / ``gt`` on the way."""
        nonlocal xt, gt, ft, have_best, best_alpha, best_f, best_x, best_g
        xt = x + a * d
        gt = np.zeros(x.size, dtype=np.float64)
        ft = problem.value_and_gradient(gt, xt)

        if ft < best_f:
            have_best = True
            best_f = ft
            best_alpha = a
            best_x = xt
            best_g = gt
        return float(np.dot(gt, d))

    a_lo = 0.0
    a_hi = 0.0
    f_lo = f0
    bracketed = False

    a_prev = 0.0
    f_prev = f0
    a = min(1.0, stp_max)

    for i in range(max_iter):
        dphi = evaluate(a)

        if ft > f0 + c1 * a * dphi0 or (i > 0 and ft >= f_prev):
            a_lo = a_prev
            f_lo = f_prev
            a_hi = a
            bracketed = True
            break

        if abs(dphi) <= -c2 * dphi0:
            return _LineSearchResult(alpha=a, xt=xt, ft=ft, gt=gt)  # strong Wolfe satisfied

        if dphi >= 0.0:
            a_lo = a
            f_lo = ft
            a_hi = a_prev
            bracketed = True
            break

        a_prev = a
        f_prev = ft

        if a >= stp_max:
            break  # cannot expand further
        a = min(2.0 * a, stp_max)

    if bracketed:
        for _ in range(max_iter):
            a = 0.5 * (a_lo + a_hi)  # bisection
            dphi = evaluate(a)

            if ft > f0 + c1 * a * dphi0 or ft >= f_lo:
                a_hi = a
            else:
                if abs(dphi) <= -c2 * dphi0:
                    return _LineSearchResult(alpha=a, xt=xt, ft=ft, gt=gt)

                if dphi * (a_hi - a_lo) >= 0.0:
                    a_hi = a_lo

                a_lo = a
                f_lo = ft

            if abs(a_hi - a_lo) < QL_EPSILON * max(1.0, abs(a)):
                break

    # strong Wolfe not reached: accept the best sufficient decrease
    if have_best:
        return _LineSearchResult(alpha=best_alpha, xt=best_x, ft=best_f, gt=best_g)
    return None


class LBFGSB(OptimizationMethod):
    """Limited-memory BFGS for bound-constrained optimization (L-BFGS-B).

    # C++ parity: ``class LBFGSB`` in ql/math/optimization/lbfgsb.hpp:35-77 (v1.43).

    Coordinates whose bound is ``+/-sys.float_info.max`` are treated as
    unbounded, so with no active bounds the method reduces to plain
    limited-memory BFGS.

    Limited memory trades accuracy of the Hessian model for cost: only the
    last ``memory`` correction pairs are stored, needing O(memory*n) storage
    and work per step instead of the O(n^2) of dense BFGS, at the price of
    linear rather than superlinear convergence.

    C++ declares two constructors — ``LBFGSB(memory, pgTol, fTol)`` and
    ``LBFGSB(lowerBound, upperBound, memory, pgTol, fTol)``. Python has no
    overloading, so the bounds arrive as keyword arguments::

        LBFGSB(10, 1e-8, 1e-9)
        LBFGSB(10, 1e-8, 1e-9, lower_bound=lo, upper_bound=hi)

    Bounds passed this way **override** the bounds of the problem's
    constraint; they are not intersected with them, so they can widen the
    feasible box as well as narrow it.
    """

    __slots__ = ("_factr", "_lower_bound", "_m", "_pg_tol", "_upper_bound")

    def __init__(
        self,
        memory: int = 10,
        pg_tol: float = 1e-8,
        f_tol: float = 1e7 * QL_EPSILON,
        *,
        lower_bound: Array | None = None,
        upper_bound: Array | None = None,
    ) -> None:
        """Build the optimizer.

        :param memory: number of stored correction pairs.
        :param pg_tol: convergence tolerance on the infinity norm of the
            projected gradient.
        :param f_tol: the iteration stops when the relative reduction of the
            objective falls below this.
        :param lower_bound: optional explicit lower bounds, overriding the
            problem's constraint. Must be given together with ``upper_bound``.
        :param upper_bound: optional explicit upper bounds.
        """
        # C++ parity: lbfgsb.cpp:487-489 and 491-497 — both constructors.
        qassert.require(memory > 0, "memory must be positive")
        qassert.require(
            (lower_bound is None) == (upper_bound is None),
            "lower and upper bounds must be supplied together",
        )
        self._m: int = memory
        self._pg_tol: float = pg_tol
        # The divide-then-multiply round trip is deliberate: ``minimize``
        # multiplies by QL_EPSILON again, and collapsing the pair to
        # ``f_tol * denom`` changes the result in the last bits.
        self._factr: float = f_tol / QL_EPSILON
        self._lower_bound: Array = (
            np.zeros(0, dtype=np.float64)
            if lower_bound is None
            else np.ascontiguousarray(lower_bound, dtype=np.float64)
        )
        self._upper_bound: Array = (
            np.zeros(0, dtype=np.float64)
            if upper_bound is None
            else np.ascontiguousarray(upper_bound, dtype=np.float64)
        )
        qassert.require(
            self._lower_bound.size == self._upper_bound.size,
            "lower and upper bound sizes are inconsistent",
        )

    def minimize(  # noqa: PLR0915 — one-to-one with the C++ function body
        self, problem: Problem, end_criteria: EndCriteria
    ) -> Type:
        """Minimize ``problem`` over the box; return the termination outcome.

        # C++ parity: lbfgsb.cpp:499-641 — ``LBFGSB::minimize``.
        """
        ec_type = Type.None_
        problem.reset()

        x: Array = problem.current_value.astype(np.float64, copy=True)
        n = int(x.size)

        lo: Array = (
            problem.constraint.lower_bound(x) if self._lower_bound.size == 0 else self._lower_bound
        )
        hi: Array = (
            problem.constraint.upper_bound(x) if self._upper_bound.size == 0 else self._upper_bound
        )

        qassert.require(
            lo.size == n and hi.size == n,
            "bounds size does not match the number of variables",
        )

        # start from a feasible point
        x = np.clip(x, lo, hi)

        g: Array = np.zeros(n, dtype=np.float64)
        f = problem.value_and_gradient(g, x)
        problem.set_current_value(x)
        problem.set_function_value(f)

        s_pairs: deque[Array] = deque()
        y_pairs: deque[Array] = deque()
        theta = 1.0
        iteration = 0
        pg_inf = 0.0  # projected-gradient infinity norm; persists for final reporting

        while True:
            # infinity norm of the projected gradient
            pg_inf = float(np.max(np.abs(np.clip(x - g, lo, hi) - x)))

            # NOTE: the problem caches the SQUARE of the projected-gradient
            # infinity norm, but EndCriteria below is handed pg_inf itself.
            problem.set_gradient_norm_value(pg_inf * pg_inf)

            if pg_inf < self._pg_tol:
                ec_type = Type.ZeroGradientNorm
                break

            hit = end_criteria.check_zero_gradient_norm(pg_inf)
            if hit is not None:
                ec_type = hit
                break

            hit = end_criteria.check_max_iterations(iteration)
            if hit is not None:
                ec_type = hit
                break

            # compact representation; drop the oldest pairs if the middle
            # matrix turns out numerically singular
            while True:
                try:
                    rep = _build_compact_rep(list(s_pairs), list(y_pairs), theta)
                    break
                # C++ catches (...) here — any failure of the middle-matrix
                # inverse means the oldest correction pair must go.
                except Exception:
                    if not s_pairs:
                        rep = _CompactRep(theta=theta)
                        break
                    s_pairs.popleft()
                    y_pairs.popleft()

            cauchy = _generalized_cauchy_point(x, g, lo, hi, rep)

            try:
                xbar = _subspace_minimization(
                    x, g, cauchy.xcp, cauchy.c, lo, hi, rep, cauchy.is_free
                )
            # C++ catches (...) here too.
            except Exception:
                xbar = cauchy.xcp  # fall back to the Cauchy point

            d: Array = xbar - x
            dphi0 = float(np.dot(g, d))

            # fall back to the (always-descent) Cauchy direction, then to the
            # projected gradient, if the subspace step is not downhill
            if float(np.linalg.norm(d)) < QL_EPSILON or dphi0 >= 0.0:
                d = cauchy.xcp - x
                dphi0 = float(np.dot(g, d))

            if float(np.linalg.norm(d)) < QL_EPSILON:
                ec_type = Type.StationaryPoint
                break

            if dphi0 >= 0.0:
                d = np.clip(x - g, lo, hi) - x
                dphi0 = float(np.dot(g, d))
                if dphi0 >= 0.0 or float(np.linalg.norm(d)) < QL_EPSILON:
                    ec_type = Type.StationaryPoint
                    break

            stp_max = _max_feasible_step(x, d, lo, hi)
            if stp_max < QL_EPSILON:
                ec_type = Type.StationaryPoint
                break

            search = _line_search_wolfe(problem, x, d, f, g, stp_max)
            if search is None:
                ec_type = Type.StationaryFunctionValue
                break

            # limited-memory update with the curvature safeguard
            s_new = search.xt - x
            y_new = search.gt - g
            sy = float(np.dot(s_new, y_new))
            yy = float(np.dot(y_new, y_new))

            if yy > 0.0 and sy > QL_EPSILON * yy:
                s_pairs.append(s_new)
                y_pairs.append(y_new)
                if len(s_pairs) > self._m:
                    s_pairs.popleft()
                    y_pairs.popleft()
                theta = yy / sy

            f_old = f
            x = search.xt
            f = search.ft
            g = search.gt
            iteration += 1

            problem.set_current_value(x)
            problem.set_function_value(f)

            # relative function-reduction stop (SciPy's factr criterion)
            denom = max(abs(f_old), abs(f), 1.0)
            if (f_old - f) <= self._factr * QL_EPSILON * denom:
                ec_type = Type.StationaryFunctionValue
                break

        problem.set_current_value(x)
        problem.set_function_value(f)
        problem.set_gradient_norm_value(pg_inf * pg_inf)
        return ec_type
