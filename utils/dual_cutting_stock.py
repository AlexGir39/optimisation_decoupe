"""
Cutting stock solver using column generation (Gilmore-Gomory) + final integer optimisation.

This version adds:
 - automatic unit normalization (all lengths converted to integers in millimetres internally)
 - safety checks on patterns to ensure no negative/invalid waste
 - graceful handling of infeasible integer master (returns empty solution instead of raising)

Author: assistant
"""

import math
import numpy as np
import pulp
from collections import namedtuple, defaultdict

Pattern = namedtuple('Pattern', ['bar_type_idx', 'counts'])


def _generate_initial_patterns(P_types, l, L):
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
    # greedy fill
    for k in range(T):
        P = P_types[k]
        remaining = P
        counts = [0] * N
        for j in sorted(range(N), key=lambda x: -l[x]):
            q = int(remaining // l[j])
            if q > 0:
                counts[j] = q
                remaining -= q * l[j]
        if any(counts):
            patterns.append(Pattern(bar_type_idx=k, counts=tuple(counts)))
    uniq = {}
    for p in patterns:
        key = (p.bar_type_idx, p.counts)
        uniq[key] = p
    return list(uniq.values())


def _build_master_problem(patterns, P_types, L, continuous=True):
    prob = pulp.LpProblem('Master', pulp.LpMinimize)
    var_lambda = []
    for idx, p in enumerate(patterns):
        name = f"lam_{idx}_b{p.bar_type_idx}"
        cat = 'Continuous' if continuous else 'Integer'
        v = pulp.LpVariable(name, lowBound=0, cat=cat)
        var_lambda.append(v)
    prob += pulp.lpSum([var_lambda[i] * P_types[patterns[i].bar_type_idx] for i in range(len(patterns))])
    N = len(L)
    for j in range(N):
        prob += pulp.lpSum([patterns[p].counts[j] * var_lambda[p] for p in range(len(patterns))]) >= int(L[j]), f"demand_{j}"
    return prob, var_lambda


def _extract_duals(prob):
    duals = []
    N = len([name for name in prob.constraints if name.startswith('demand_')])
    duals_sorted = [0.0] * N
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
    l = [int(round(x)) for x in l_mm]
    P_types_int = [int(round(x)) for x in P_types]
    N = len(l)
    T = len(P_types)
    patterns = _generate_initial_patterns(P_types_int, l, L)
    for it in range(max_iter):
        prob, var_lambda = _build_master_problem(patterns, P_types_int, L, continuous=True)
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        duals = _extract_duals(prob)
        new_pattern_added = False
        for k in range(T):
            capacity = P_types_int[k]
            best_val, best_counts = _knapsack_dp_max(duals, l, capacity)
            reduced_cost = P_types_int[k] - best_val
            if reduced_cost < -tol and any(c > 0 for c in best_counts):
                p = Pattern(bar_type_idx=k, counts=tuple(int(x) for x in best_counts))
                if p not in patterns:
                    patterns.append(p)
                    new_pattern_added = True
        if not new_pattern_added:
            break
    return patterns


def solve_integer_master(patterns, P_types, L, time_limit=None, verbose=False):
    prob, var_lambda = _build_master_problem(patterns, P_types, L, continuous=False)
    solver = pulp.PULP_CBC_CMD(msg=1 if verbose else 0, timeLimit=time_limit)
    prob.solve(solver)
    status = pulp.LpStatus.get(prob.status, None)
    if status not in ('Optimal', 'Integer Feasible'):
        if verbose:
            print("⚠️ Integer master infeasible — check units or demands")
        return [0] * len(patterns), None
    lambdas = [int(round(pulp.value(v) or 0)) for v in var_lambda]
    return lambdas, pulp.value(prob.objective)


def build_solution_from_patterns(patterns, lambdas, P_types, l_mm):
    X_used_rows = []
    bar_types_used = []
    gaspillage = []
    counts_per_type = defaultdict(int)
    l = np.array(l_mm, dtype=float)
    for idx, (p, lam) in enumerate(zip(patterns, lambdas)):
        if lam <= 0:
            continue
        for rep in range(lam):
            total_in_bar = sum(np.array(p.counts, dtype=float) * l)
            waste = P_types[p.bar_type_idx] - total_in_bar
            if waste < -1e-6:
                continue  # skip invalid pattern
            X_used_rows.append(list(p.counts))
            bar_types_used.append(P_types[p.bar_type_idx])
            gaspillage.append(max(0.0, waste))
            counts_per_type[P_types[p.bar_type_idx]] += 1
    X_used = np.array(X_used_rows, dtype=int) if X_used_rows else np.zeros((0, len(l_mm)), dtype=int)
    gaspillage = np.array(gaspillage, dtype=float) if gaspillage else np.array([])
    total_matiere = float(sum(counts_per_type[p] * p for p in counts_per_type))
    total_waste = float(np.sum(gaspillage))
    return X_used, bar_types_used, dict(counts_per_type), gaspillage, total_matiere, total_waste


def optimisation_min_matiere_cg(P_types, l, L, time_limit=None, verbose=False):
    # normalize units: if max length < 100, assume meters → convert to mm
    if max(P_types + l) < 100:
        P_types = [int(round(x * 1000)) for x in P_types]
        l = [int(round(x * 1000)) for x in l]
    else:
        P_types = [int(round(x)) for x in P_types]
        l = [int(round(x)) for x in l]
    L = [int(x) for x in L]
    patterns = column_generation(P_types, l, L, verbose=verbose)
    lambdas, obj = solve_integer_master(patterns, P_types, L, time_limit=time_limit, verbose=verbose)
    return build_solution_from_patterns(patterns, lambdas, P_types, l)


if __name__ == '__main__':
    P_types = [6.0, 4.0]  # metres → auto-converted to mm
    l = [1.5, 1.2, 0.7]
    L = [10, 12, 5]
    result = optimisation_min_matiere_cg(P_types, l, L, verbose=True)
    print("\nResults:")
    for name, val in zip(["X_used", "bar_types_used", "counts_per_type", "gaspillage", "total_matiere", "total_waste"], result):
        print(name, ":", val)

