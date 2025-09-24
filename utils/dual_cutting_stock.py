"""
Cutting stock solver using column generation (Gilmore-Gomory) + final integer optimisation.

Functions:
 - optimisation_min_matiere_cg(P_types, l_mm, L, time_limit=None, verbose=False)

Outputs: same format as your original function:
  X_used, bar_types_used, counts_per_type, gaspillage_par_barre, total_matiere, total_waste

Requires: pulp, numpy

Notes:
 - lengths of demanded pieces `l_mm` are expected in millimetres (like your original code).
 - P_types are lengths of stock bars (same units as l_mm but usually in mm or metres -- we keep units consistent).
 - The algorithm works in two phases: solve continuous master LP by column generation, then solve an IP on the generated columns.
 - The knapsack subproblem is solved by integer dynamic programming (exact) for each stock type.

Author: assistant
"""

import math
import numpy as np
import pulp
from collections import namedtuple, defaultdict

Pattern = namedtuple('Pattern', ['bar_type_idx', 'counts'])


def _generate_initial_patterns(P_types, l, L):
    """Create a set of simple initial patterns: for each piece type j, pack as many as possible into each bar type."""
    patterns = []
    T = len(P_types)
    N = len(l)
    for k in range(T):
        P = P_types[k]
        for j in range(N):
            max_q = int(P // l[j])
            if max_q <= 0:
                continue
            counts = [0] * N
            counts[j] = max_q
            patterns.append(Pattern(bar_type_idx=k, counts=tuple(counts)))
    # Also add a greedy-fill pattern for each bar type: fill with largest pieces first
    for k in range(T):
        P = P_types[k]
        remaining = P
        counts = [0] * N
        # sort by length descending
        for j in sorted(range(N), key=lambda x: -l[x]):
            q = int(remaining // l[j])
            if q > 0:
                counts[j] = q
                remaining -= q * l[j]
        if any(counts):
            patterns.append(Pattern(bar_type_idx=k, counts=tuple(counts)))
    # Deduplicate
    uniq = {}
    for p in patterns:
        key = (p.bar_type_idx, p.counts)
        uniq[key] = p
    return list(uniq.values())


def _build_master_problem(patterns, P_types, L, continuous=True):
    """Build a PuLP LP/MIP for the master problem given a list of patterns.
    Returns (prob, var_lambda, dual_vars_holder)
    If continuous=True, lambda are continuous >=0; otherwise Integer.
    dual_vars_holder is list of pulp.LpVariable objects for constraints (useful to extract duals).
    """
    prob = pulp.LpProblem('Master', pulp.LpMinimize)
    # variables lambda_p
    var_lambda = []
    for idx, p in enumerate(patterns):
        name = f"lam_{idx}_b{p.bar_type_idx}"
        if continuous:
            v = pulp.LpVariable(name, lowBound=0, cat='Continuous')
        else:
            v = pulp.LpVariable(name, lowBound=0, cat='Integer')
        var_lambda.append(v)
    # objective: sum lambda_p * P_types[bar_type_idx]
    prob += pulp.lpSum([var_lambda[i] * P_types[patterns[i].bar_type_idx] for i in range(len(patterns))])

    # constraints: for each piece type j, sum_p a_{j,p} * lambda_p >= L_j
    N = len(L)
    dual_holders = []
    for j in range(N):
        cons = pulp.lpSum([patterns[p].counts[j] * var_lambda[p] for p in range(len(patterns))]) >= int(L[j])
        c = prob.addConstraint(cons, name=f"demand_{j}")
        dual_holders.append(c)

    return prob, var_lambda, dual_holders


def _extract_duals(prob, continuous=True):
    """Extract duals (shadow prices) for demand constraints from a solved PuLP problem.
    PuLP exposes duals via .constraints[name].pi when using CBC in recent versions.
    We'll be robust and try multiple access patterns.
    """
    duals = []
    # constraints in prob.constraints are in dict
    for name, cons in prob.constraints.items():
        # demand constraints were named demand_{j}
        if name.startswith('demand_'):
            # try to get Pi / shadow price
            pi = None
            # older/newer pulp versions: try attribute 'pi' or 'pi' in dict-like
            pi = getattr(cons, 'pi', None)
            if pi is None:
                # some versions expose as pulp.value(cons.pi)
                try:
                    pi = pulp.value(cons.pi)
                except Exception:
                    pi = None
            if pi is None:
                # As last resort try to read from solver status object via prob.solution, but skip
                pi = 0.0
            duals.append(float(pi))
    # order by demand_0, demand_1, ... ensure correct order
    duals_sorted = [0.0] * len(duals)
    for name, cons in prob.constraints.items():
        if name.startswith('demand_'):
            j = int(name.split('_')[1])
            pi = getattr(cons, 'pi', None)
            if pi is None:
                try:
                    pi = pulp.value(cons.pi)
                except Exception:
                    pi = 0.0
            duals_sorted[j] = float(pi or 0.0)
    return duals_sorted


def _knapsack_dp_max(profits, weights, capacity):
    """Integer knapsack DP returning (best_value, counts tuple)
    - profits: list of profit per item type (float)
    - weights: list of weight per item type (float)
    - capacity: scalar (float)

    Items can be used multiple times (unbounded knapsack) -> this is unbounded integer knapsack.
    We'll solve by DP on discretized units: but lengths are floats. To avoid precision issues,
    we will work with integer units by dividing by gcd of weights quantized to mm (or smallest unit).

    For correctness and reasonable speed we assume weights are rational and multiplied to integers outside.
    """
    # Here, assume weights are integer (e.g., mm). If floats, caller should convert.
    cap = int(capacity)
    N = len(weights)
    dp = [(-1e9, None)] * (cap + 1)
    dp[0] = (0.0, (0,) * N)
    for c in range(1, cap + 1):
        best = (-1e9, None)
        for j in range(N):
            w = int(weights[j])
            if w <= 0 or w > c:
                continue
            prev_val, prev_counts = dp[c - w]
            if prev_counts is None:
                continue
            val = prev_val + profits[j]
            if val > best[0]:
                counts = list(prev_counts)
                counts[j] += 1
                best = (val, tuple(counts))
        dp[c] = best
    best_val, best_counts = max(dp, key=lambda x: x[0])
    if best_counts is None:
        return 0.0, tuple([0] * N)
    return best_val, best_counts


def column_generation(P_types, l_mm, L, max_iter=200, tol=1e-5, verbose=False):
    """Perform column generation. Returns a list of patterns (Pattern objects).
    l_mm: lengths in millimetres (integers)
    P_types: bar lengths in same units
    L: demand integers
    """
    # ensure integer weights for DP
    l = [int(round(x)) for x in l_mm]
    P_types_int = [int(round(x)) for x in P_types]
    N = len(l)
    T = len(P_types)

    patterns = _generate_initial_patterns(P_types_int, l, L)
    if verbose:
        print(f"Initial patterns: {len(patterns)}")

    for it in range(max_iter):
        # build and solve LP master
        prob, var_lambda, dual_holders = _build_master_problem(patterns, P_types_int, L, continuous=True)
        # Solve with CBC
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        status = pulp.LpStatus.get(prob.status, None)
        if verbose:
            print(f"Master LP iter {it}: status {status}, obj {pulp.value(prob.objective)}")

        # extract duals
        # Try to use constraint.pi; if not available, fallback to solving with pulp. But we'll try
        duals = _extract_duals(prob, continuous=True)
        if verbose:
            print("Duals:", duals)

        # Solve knapsack (pricing) for each bar type to find a column with negative reduced cost
        new_pattern_added = False
        for k in range(T):
            capacity = P_types_int[k]
            # profits are duals (pi_j) for each piece j
            profits = duals
            weights = l
            best_val, best_counts = _knapsack_dp_max(profits, weights, capacity)
            # reduced cost = bar_length - sum(pi_j * x_j) ; we seek < 0 to improve
            reduced_cost = P_types_int[k] - best_val
            if verbose:
                print(f"  bar type {k} cap {capacity}: best profit {best_val}, red_cost {reduced_cost}")
            # add pattern if reduced cost < -tol and not trivial (not all zeros)
            if reduced_cost < -tol and any(c > 0 for c in best_counts):
                p = Pattern(bar_type_idx=k, counts=tuple(int(x) for x in best_counts))
                if p not in patterns:
                    patterns.append(p)
                    new_pattern_added = True
                    if verbose:
                        print(f"    adding pattern for bar {k}: {p.counts}")
        if not new_pattern_added:
            if verbose:
                print("No improving pattern found — optimal LP master reached.")
            break
    return patterns


def solve_integer_master(patterns, P_types, L, time_limit=None, verbose=False):
    """Given a set of patterns, solve final integer master (MIP) to obtain integer counts of patterns.
    Returns lambdas (list) and objective.
    """
    prob, var_lambda, _ = _build_master_problem(patterns, P_types, L, continuous=False)
    # optionally set solver time limit if needed
    solver = pulp.PULP_CBC_CMD(msg=1 if verbose else 0, timeLimit=time_limit)
    prob.solve(solver)
    status = pulp.LpStatus.get(prob.status, None)
    if verbose:
        print("Integer master status:", status)
    if status not in ('Optimal', 'Integer Feasible'):
        raise RuntimeError('Integer master did not find a feasible/optimal solution: ' + str(status))
    lambdas = [int(round(pulp.value(v) or 0)) for v in var_lambda]
    return lambdas, pulp.value(prob.objective)


def build_solution_from_patterns(patterns, lambdas, P_types, l_mm):
    """Convert pattern counts to the same outputs your original function returned.
    - X_used: each used pattern becomes a row with counts of pieces
    - bar_types_used: stock length for each used pattern
    - counts_per_type: dict {P_k: number_of_bars_used}
    - gaspillage_par_barre: list of waste per used bar
    - total_matiere, total_waste
    """
    used_rows = []
    bar_types_used = []
    gaspillage = []
    X_used_rows = []
    counts_per_type = defaultdict(int)
    l = np.array(l_mm, dtype=float)

    for idx, (p, lam) in enumerate(zip(patterns, lambdas)):
        if lam <= 0:
            continue
        for rep in range(lam):
            X_used_rows.append(list(p.counts))
            bar_types_used.append(P_types[p.bar_type_idx])
            total_in_bar = sum(np.array(p.counts, dtype=float) * l)
            gaspillage.append(P_types[p.bar_type_idx] - total_in_bar)
            counts_per_type[P_types[p.bar_type_idx]] += 1

    X_used = np.array(X_used_rows, dtype=int) if len(X_used_rows) else np.zeros((0, len(l_mm)), dtype=int)
    gaspillage = np.array(gaspillage, dtype=float) if len(gaspillage) else np.array([])
    total_matiere = float(sum(counts_per_type[p] * p for p in counts_per_type))
    total_waste = float(np.sum(gaspillage))
    return X_used, bar_types_used, dict(counts_per_type), gaspillage, total_matiere, total_waste


def optimisation_min_matiere_cg(P_types, l, L, time_limit=None, verbose=False):
    """Full pipeline: column generation + integer master solve.

    Inputs:
      P_types : list of stock lengths (same unit as l). Can be floats but are converted to integers (rounded).
      l       : list of piece lengths (same unit as P_types). If your input is in mm, pass mm integers.
      L       : list of integer demands

    Returns:
      X_used, bar_types_used, counts_per_type, gaspillage_par_barre, total_matiere, total_waste
    """
    # convert to arrays and integer mm units
    l_mm = [int(round(x)) for x in l]
    P_types_int = [int(round(x)) for x in P_types]
    L_arr = [int(x) for x in L]

    if verbose:
        print("Starting column generation...")
    patterns = column_generation(P_types_int, l_mm, L_arr, verbose=verbose)
    if verbose:
        print(f"Generated {len(patterns)} patterns. Now solving integer master...")

    lambdas, obj = solve_integer_master(patterns, P_types_int, L_arr, time_limit=time_limit, verbose=verbose)

    # Build outputs in original unit (we keep lengths in same unit as input)
    X_used, bar_types_used, counts_per_type, gaspillage, total_matiere, total_waste = build_solution_from_patterns(patterns, lambdas, P_types_int, l_mm)

    return X_used, bar_types_used, counts_per_type, gaspillage, total_matiere, total_waste
