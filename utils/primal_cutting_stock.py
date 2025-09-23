import pulp
import numpy as np
from math import floor

def optimisation_min_matiere(P_types, l, L, solver_msg=False):
    """
    P_types : liste des longueurs disponibles des barres (types) [P1,P2,...,PT]
    l       : liste des longueurs des pièces demandées [l1,...,lN]
    L       : quantités demandées [L1,...,LN]
    Retourne :
      - X_used : matrice (#barres_utilisees x N) avec nombre de pieces j dans chaque barre utilisée
      - bar_types_used : liste des longueurs de barre attribuées à chaque ligne utilisée
      - counts_per_type : dict {P_k: nombre_de_barres_utilisees}
      - gaspillage_par_barre : liste des gaspillage par barre
      - total_matiere : somme des longueurs de barres utilisées (objectif)
      - total_waste : gaspillage total
    """
    P_types = list(map(float, P_types))
    l = np.array(l, dtype=float)/1000
    L = np.array(L, dtype=int)
    T = len(P_types)
    N = len(l)

    # Bornes : au pire une pièce par barre
    M = int(np.sum(L))

    prob = pulp.LpProblem("CuttingStock_min_matiere", pulp.LpMinimize)

    # Variables
    X = [[pulp.LpVariable(f"X_{i}_{j}", lowBound=0, cat="Integer") for j in range(N)] for i in range(M)]
    Y = [[pulp.LpVariable(f"Y_{i}_{k}", cat="Binary") for k in range(T)] for i in range(M)]

    # Objectif : minimiser la longueur totale de barres utilisées
    prob += pulp.lpSum([P_types[k] * Y[i][k] for i in range(M) for k in range(T)])

    # Contraintes de demande
    for j in range(N):
        prob += pulp.lpSum([X[i][j] for i in range(M)]) == int(L[j])

    # Chaque barre a au plus un type
    for i in range(M):
        prob += pulp.lpSum([Y[i][k] for k in range(T)]) <= 1

    # Capacité : somme des longueurs de pièces <= longueur du type choisi
    for i in range(M):
        prob += pulp.lpSum([X[i][j] * l[j] for j in range(N)]) <= \
                pulp.lpSum([P_types[k] * Y[i][k] for k in range(T)])

    # Si barre non utilisée, pas de pièces dedans
    for i in range(M):
        for j in range(N):
            prob += X[i][j] <= L[j] * pulp.lpSum([Y[i][k] for k in range(T)])

    # Symmetry-breaking (optionnel mais utile) : les barres utilisées contiguës
    for i in range(M-1):
        prob += pulp.lpSum([Y[i][k] for k in range(T)]) >= pulp.lpSum([Y[i+1][k] for k in range(T)])

    # Résoudre
    prob.solve(pulp.PULP_CBC_CMD(msg=1 if solver_msg else 0))

    if pulp.LpStatus[prob.status] != "Optimal":
        return None, None, None, None, None, None

    # Extraction : lignes utilisées (où sum_k Y[i,k] == 1)
    used_rows = []
    bar_types_used = []
    for i in range(M):
        val_sum = sum(int(round(pulp.value(Y[i][k]) or 0)) for k in range(T))
        if val_sum == 1:
            chosen_k = next(k for k in range(T) if int(round(pulp.value(Y[i][k]) or 0)) == 1)
            used_rows.append(i)
            bar_types_used.append(P_types[chosen_k])

    # Construire X_used, gaspillage et compter par type
    X_used = []
    gaspillage = []
    for i in used_rows:
        row = [int(round(pulp.value(X[i][j]) or 0)) for j in range(N)]
        X_used.append(row)
        total_in_bar = sum(row[j] * l[j] for j in range(N))
        chosen_k = next(k for k in range(T) if int(round(pulp.value(Y[i][k]) or 0)) == 1)
        P_assigned = P_types[chosen_k]
        gaspillage.append(P_assigned - total_in_bar)

    X_used = np.array(X_used, dtype=int) if len(X_used) > 0 else np.zeros((0, N), dtype=int)
    gaspillage = np.array(gaspillage, dtype=float) if len(gaspillage) > 0 else np.array([])

    counts_per_type = {P_types[k]: sum(int(round(pulp.value(Y[i][k]) or 0)) for i in range(M)) for k in range(T)}
    total_matiere = float(sum(counts_per_type[p] * p for p in counts_per_type))
    total_waste = float(np.sum(gaspillage))

    return X_used, bar_types_used, counts_per_type, gaspillage, total_matiere, total_waste