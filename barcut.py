import streamlit as st
import pulp
import numpy as np
import pandas as pd

def optimisation_coupe(P, L, l):
    '''
    P : liste des longueurs de barres disponibles
    L : liste des quantités demandées pour chaque longueur de pièce
    l : liste des longueurs des pièces demandées
    Retourne le plan de coupe optimal X (M x N), le gaspillage par barre et le gaspillage total
    '''
    P = np.array(P)
    L = np.array(L)
    l = np.array(l)
    M = len(P)
    N = len(L)

    # Création du problème d'optimisation linéaire en nombres entiers
    prob = pulp.LpProblem("Probleme_de_coupe", pulp.LpMinimize)

    # Variables de décision : X[i,j] = nombre de pièces de type j coupées dans la barre i
    X = [[pulp.LpVariable(f"X_{i}_{j}", lowBound=0, cat="Integer") for j in range(N)] for i in range(M)]

    # Objectif : minimiser le gaspillage total
    prob += pulp.lpSum([P[i] - pulp.lpSum([X[i][j]*l[j] for j in range(N)]) for i in range(M)])

    # Contraintes : satisfaire la demande pour chaque type de pièce
    for j in range(N):
        prob += pulp.lpSum([X[i][j] for i in range(M)]) == L[j]

    # Contraintes : ne pas dépasser la longueur de chaque barre
    for i in range(M):
        prob += pulp.lpSum([X[i][j]*l[j] for j in range(N)]) <= P[i]

    # Résolution avec CBC
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != "Optimal":
        return None, None, None

    # Extraction du plan de coupe
    X_sol = np.zeros((M, N), dtype=int)
    for i in range(M):
        for j in range(N):
            X_sol[i, j] = int(pulp.value(X[i][j]))

    # Calcul du gaspillage
    utilise_par_barre = np.sum(X_sol * l, axis=1)
    gaspillage_par_barre = P - utilise_par_barre
    gaspillage_total = np.sum(gaspillage_par_barre)

    return X_sol, gaspillage_par_barre, gaspillage_total


# --- Interface Streamlit ---
st.title("🪚 Optimiseur de plan de coupe")

st.markdown("Cette application résout un problème de **découpe de barres** pour minimiser le gaspillage.")

# Entrées utilisateur
st.sidebar.header("Paramètres d'entrée")

# Longueurs des barres disponibles
P_str = st.sidebar.text_input("Longueurs des barres disponibles (séparées par des virgules)", "6,6,6")
P = list(map(float, P_str.split(",")))

# Longueurs des pièces demandées
l_str = st.sidebar.text_input("Longueurs des pièces demandées (séparées par des virgules)", "1,5")
l = list(map(float, l_str.split(",")))

# Tableau interactif pour les quantités demandées
st.subheader("📋 Quantités demandées pour chaque longueur")
df_input = pd.DataFrame({
    "Longueur demandée": l,
    "Quantité demandée": [1 for _ in l]  # valeur par défaut
})

df_result = st.data_editor(df_input, num_rows="fixed")
L = df_result["Quantité demandée"].tolist()

# Vérification cohérence
if len(L) != len(l):
    st.error("❌ La taille du vecteur L doit correspondre à la taille du vecteur l.")
else:
    if st.button("Optimiser"):
        X_sol, gaspillage_par_barre, gaspillage_total = optimisation_coupe(P, L, l)

        if X_sol is None:
            st.error("⚠️ Aucun plan optimal trouvé.")
        else:
            st.success("✅ Solution optimale trouvée !")

            # Affichage du plan de coupe
            df_plan = pd.DataFrame(X_sol,
                                   columns=[f"Longueur {li}m" for li in l],
                                   index=[f"Barre {p}m" for p in P])
            st.subheader("📊 Plan de coupe optimal")
            st.dataframe(df_plan)

            # Gaspillage par barre
            df_gaspillage = pd.DataFrame({
                "Barre": [f"Barre {i+1}" for i in range(len(P))],
                "Gaspillage": gaspillage_par_barre
            })
            st.subheader("🗑️ Gaspillage par barre")
            st.table(df_gaspillage)

            # Gaspillage total
            st.metric("Gaspillage total", f"{gaspillage_total}")
