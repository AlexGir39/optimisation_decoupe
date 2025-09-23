import pulp
import numpy as np

# ----------------------
# utilitaires : DP knapsack non borné
# ----------------------
def unbounded_knapsack_dp(values, weights, capacity):
    """
    DP pour knapsack UNBOUNDED.
    values: list[float] (profits)
    weights: list[int] (poids entiers)
    capacity: int (entier)
    retourne (best_value, counts_list)
    """
    n = len(values)
    # dp[c] = best profit achievable with capacity exactly c
    NEG = -1e90
    dp = [NEG] * (capacity + 1)
    back = [-1] * (capacity + 1)
    dp[0] = 0.0
    for c in range(capacity + 1):
        if dp[c] < -1e80:
            continue
        for i in range(n):
            w = weights[i]
            if w <= 0 or c + w > capacity:
                continue
            val = dp[c] + values[i]
            if val > dp[c + w]:
                dp[c + w] = val
                back[c + w] = i
    # choose best over all <= capacity
    best_c = max(range(capacity + 1), key=lambda cc: dp[cc])
    best_val = dp[best_c]
    # reconstruct counts
    counts = [0] * n
    c = best_c
    while c > 0 and back[c] != -1:
        idx = back[c]
        counts[idx] += 1
        c -= weights[idx]
    return best_val, counts

# ----------------------
# Master LP solver (retourne z_vals et duals)
# ----------------------
def solve_master_lp(patterns, L, solver):
    """
    patterns: list of dict {'a': [a_j], 'type': k, 'cost': cost}
    L: list demandes
    solver: pulp solver instance that returns duals (GLPK recommended)
    retourne (status, z_vals, duals_list)
    """
    prob = pulp.LpProblem("MasterLP", pulp.LpMinimize)
    z_vars = [pulp.LpVariable(f"z_{p}", lowBound=0, cat='Continuous') for p in range(len(patterns))]
    prob += pulp.lpSum([patterns[p]['cost'] * z_vars[p] for p in range(len(patterns))])
    # contraintes de demande
    for j in range(len(L)):
        prob += pulp.lpSum([patterns[p]['a'][j] * z_vars[p] for p in range(len(patterns))]) >= L[j], f"Demand_{j}"
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    z_vals = [pulp.value(v) for v in z_vars]
    # récupère les duaux
    duals = None
    try:
        duals = [prob.constraints[f"Demand_{j}"].pi for j in range(len(L))]
    except Exception:
        duals = None
    return status, z_vals, duals

# ----------------------
# generate simple initial patterns
# ----------------------
def generate_initial_patterns(P_mm, l_mm):
    patterns = []
    T = len(P_mm)
    N = len(l_mm)
    for k, Pk in enumerate(P_mm):
        for j, lj in enumerate(l_mm):
            q = Pk // lj
            if q >= 1:
                a = [0]*N
                a[j] = int(q)
                patterns.append({'a': a, 'type': k, 'cost': Pk})
    # fallback : empty patterns (one per type)
    if not patterns:
        for k, Pk in enumerate(P_mm):
            patterns.append({'a': [0]*N, 'type': k, 'cost': Pk})
    return patterns

# ----------------------
# Column generation + integer master
# ----------------------
def optimisation_min_matiere_dual(P_types, l, L, max_iter=200, tol=1e-6, verbose=False):
    """
    Dualized / column generation version that minimizes total material used.
    Inputs:
      - P_types : list of bar lengths in meters (floats)  (ex: [6,8])
      - l       : list of piece lengths in millimeters (ints or floats) (ex: [1000, 1100])
      - L       : list of demands (ints)
    Returns (same outputs as optimisation_min_matiere):
      X_used (array), bar_types_used (list), counts_per_type (dict), gaspillage (list), total_matiere (float, meters), total_waste (float, meters)
    Requirements: GLPK recommended (for duals). If GLPK absent, function will try CBC but may fail to generate columns.
    """
    # --- normalize units: convert everything to millimeters ints internally
    P_types = [float(x) for x in P_types]  # meters
    P_mm = [int(round(1000.0 * p)) for p in P_types]   # convert to mm
    l_mm = [int(round(x)) for x in l]                  # assume l given in mm
    L = [int(x) for x in L]

    T = len(P_mm)
    N = len(l_mm)
    if N == 0:
        return np.zeros((0,0), dtype=int), [], {pt:0 for pt in P_types}, np.array([]), 0.0, 0.0

    # initial patterns
    patterns = generate_initial_patterns(P_mm, l_mm)

    # choose solver for LP master
    # try GLPK first (returns duals)
    try:
        glpk = pulp.GLPK_CMD(msg=0)
        status, z_vals, duals = solve_master_lp(patterns, L, solver=glpk)
        if duals is None:
            # fallback warning
            if verbose:
                print("GLPK did not return duals.")
    except Exception as e:
        if verbose:
            print("GLPK not available, falling back to CBC for LP (may not return duals):", e)
        status, z_vals, duals = solve_master_lp(patterns, L, solver=pulp.PULP_CBC_CMD(msg=0))

    if duals is None:
        # cannot proceed with column generation reliably
        raise RuntimeError("Le solveur LP n'a pas renvoyé les multiplicateurs duals. Installez GLPK ou un solveur LP qui fournit les duals.")

    # column generation loop
    it = 0
    while it < max_iter:
        it += 1
        if verbose:
            print(f"it {it}, duals = {duals}")

        added = False
        # pour chaque type de barre, résoudre knapsack unbounded
        for k, Pk in enumerate(P_mm):
            best_val, counts = unbounded_knapsack_dp(duals, l_mm, Pk)
            # best_val is sum_j pi_j * a_j
            if best_val is None:
                continue
            reduced_cost = Pk - best_val
            if best_val > Pk + tol:
                # add pattern
                a = [int(c) for c in counts]
                patterns.append({'a': a, 'type': k, 'cost': Pk})
                added = True
                if verbose:
                    print("Added pattern for type", k, "counts", a, "best_val", best_val)
        if not added:
            break
        # resolve master LP
        status, z_vals, duals = solve_master_lp(patterns, L, solver=glpk)

    # integer master with generated patterns
    prob_int = pulp.LpProblem("MasterInteger", pulp.LpMinimize)
    z_int_vars = [pulp.LpVariable(f"z_{p}", lowBound=0, cat='Integer') for p in range(len(patterns))]
    prob_int += pulp.lpSum([patterns[p]['cost'] * z_int_vars[p] for p in range(len(patterns))])
    for j in range(len(L)):
        prob_int += pulp.lpSum([patterns[p]['a'][j] * z_int_vars[p] for p in range(len(patterns))]) >= L[j]
    prob_int.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob_int.status] != 'Optimal':
        # fallback: round up LP solution
        z_int = {i: int(np.ceil(z_vals[i] if z_vals[i] is not None else 0)) for i in range(len(patterns))}
    else:
        z_int = {i: int(pulp.value(z_int_vars[i])) for i in range(len(patterns))}

    # expand integer solution into per-bar allocations
    X_used = []
    bar_types_used = []
    waste_per_bar = []
    counts_per_type = {pt: 0 for pt in P_types}
    for p_idx, pat in enumerate(patterns):
        count = z_int.get(p_idx, 0)
        for _ in range(count):
            X_used.append(pat['a'])
            bar_types_used.append(P_types[pat['type']])  # in meters
            counts_per_type[P_types[pat['type']]] += 1
            used_len_mm = sum(pat['a'][j] * l_mm[j] for j in range(N))
            waste_per_bar.append((pat['cost'] - used_len_mm)/1000.0)  # convert to meters

    if len(X_used) == 0:
        X_used = np.zeros((0, N), dtype=int)
    else:
        X_used = np.array(X_used, dtype=int)
    waste_per_bar = np.array(waste_per_bar, dtype=float) if len(waste_per_bar) > 0 else np.array([])

    total_matiere = sum(bar_types_used)  # already in meters
    total_waste = float(np.sum(waste_per_bar))  # in meters

    return X_used, bar_types_used, counts_per_type, waste_per_bar, total_matiere, total_waste
