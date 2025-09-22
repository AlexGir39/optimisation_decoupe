import streamlit as st
import pulp
import numpy as np
import pandas as pd

# --- Fonction d'optimisation Cutting Stock ---
def optimisation_coupe_min_barres(P_types, l, L):
    """
    P_types : liste des longueurs disponibles des barres (types)
    l       : longueurs des pièces demandées
    L       : quantités demandées pour chaque pièce
    Retourne X_sol, nombre de barres utilisées de chaque type, gaspillage par barre, gaspillage total
    """
    P_types = np.array(P_types)
    l = np.array(l)
    L = np.array(L)
    N_pieces = len(l)
    N_barres = len(P_types)

    # On suppose un nombre maximal de barres assez grand
    max_barres = sum(L)  # au pire, une pièce par barre
    M = max_barres

    # Variables de décision :
    # X[i,j] = nombre de pièces de type j dans la barre i
    X = [[pulp.LpVariable(f"X_{i}_{j}", lowBound=0, cat="Integer") for j in range(N_pieces)] for i in range(M)]
    # Y[i] = type de la barre i utilisée (0 = pas utilisée, 1 = utilisée)
    Y = [pulp.LpVariable(f"Y_{i}", cat="Binary") for i in range(M)]
    # type_barre[i] = type de barre choisie pour la barre i
    type_barre = [pulp.LpVariable(f"T_{i}", lowBound=0, upBound=N_barres-1, cat="Integer") for i in range(M)]  # relaxation integer

    prob = pulp.LpProblem("CuttingStock_MinBarres", pulp.LpMinimize)

    # Objectif : minimiser le nombre de barres utilisées
    prob += pulp.lpSum(Y)

    # Contraintes : satisfaire la demande pour chaque type de pièce
    for j in range(N_pieces):
        prob += pulp.lpSum([X[i][j] for i in range(M)]) == L[j]

    # Contraintes : ne pas dépasser la longueur de la barre utilisée
    for i in range(M):
        # On choisit une barre parmi les types disponibles
        # Approximation : toutes les barres ont même longueur max(P_types)
        prob += pulp.lpSum([X[i][j]*l[j] for j in range(N_pieces)]) <= pulp.lpSum([Y[i]*P_types[k] for k in range(N_barres)])

    # Si aucune pièce dans la barre, Y[i] = 0
    for i in range(M):
        for j in range(N_pieces):
            prob += X[i][j] <= L[j]*Y[i]

    # Résolution
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # Extraction solution
    X_sol = np.zeros((M, N_pieces), dtype=int)
    Y_sol = np.zeros(M, dtype=int)
    for i in range(M):
        Y_sol[i] = int(pulp.value(Y[i]))
        for j in range(N_pieces):
            X_sol[i, j] = int(pulp.value(X[i][j]))

    # Gaspillage
    P_assigned = []
    gaspillage = []
    for i in range(M):
        if Y_sol[i] == 1:
            # on choisit la barre la plus adaptée pour cette ligne
            total_piece = sum(X_sol[i]*l)
            best_barre = min([p for p in P_types if p >= total_piece])
            P_assigned.append(best_barre)
            gaspillage.append(best_barre - total_piece)
    gaspillage_total = sum(gaspillage)

    # On retourne seulement les lignes utilisées
    X_sol = X_sol[Y_sol==1]
    return X_sol, P_assigned, gaspillage, gaspillage_total

# --- Streamlit ---
st.title("🪚 Optimiseur Cutting Stock minimal barres")

st.markdown("Donnez les longueurs des barres disponibles. L'algorithme choisira le nombre minimal de barres pour satisfaire la demande.")

# Entrée des barres
P_types_str = st.sidebar.text_input("Longueurs de barres disponibles (séparées par des virgules)", "6,8")
P_types = list(map(float, P_types_str.split(",")))

# Entrée des pièces
l_str = st.sidebar.text_input("Longueurs des pièces demandées (séparées par des virgules)", "1,5")
l = list(map(float, l_str.split(",")))

# Tableau interactif des quantités demandées
st.subheader("📋 Quantités demandées pour chaque longueur de pièce")
df_input = pd.DataFrame({
    "Longueur demandée": l,
    "Quantité demandée": [1 for _ in l]
})
df_result = st.data_editor(df_input, num_rows="fixed")
L = df_result["Quantité demandée"].tolist()

if st.button("Optimiser"):
    X_sol, P_assigned, gaspillage, gaspillage_total = optimisation_coupe_min_barres(P_types, l, L)
    if X_sol is None or len(X_sol) == 0:
        st.error("⚠️ Aucun plan optimal trouvé.")
    else:
        st.success(f"✅ Solution trouvée avec {len(P_assigned)} barres utilisées !")

        # Plan de coupe
        df_plan = pd.DataFrame(X_sol,
                               columns=[f"Longueur {li}" for li in l],
                               index=[f"Barre {i+1} ({P_assigned[i]}m)" for i in range(len(P_assigned))])
        st.subheader("📊 Plan de coupe optimal")
        st.dataframe(df_plan)

        # Gaspillage
        df_gaspillage = pd.DataFrame({
            "Barre": [f"Barre {i+1}" for i in range(len(P_assigned))],
            "Longueur de barre": P_assigned,
            "Gaspillage": gaspillage
        })
        st.subheader("🗑️ Gaspillage par barre")
        st.table(df_gaspillage)

        # Total
        st.metric("Gaspillage total", f"{gaspillage_total}")
