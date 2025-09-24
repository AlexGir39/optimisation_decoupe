from utils.primal_cutting_stock import optimisation_min_matiere
from utils.dual_cutting_stock import optimisation_min_matiere_cg

import streamlit as st
import pulp
import numpy as np
import pandas as pd

# --- Streamlit ---
st.title("🪚 Optimiseur Cutting Stock minimal matière")

st.markdown("Donnez les longueurs des barres disponibles. L'algorithme choisira automatiquement "
            "le nombre et le type de barres pour minimiser la matière utilisée.")

# Entrée des barres
P_types_str = st.sidebar.text_input("Longueurs de barres disponibles [en mètres, séparées par virgule]", "6")
P_types = list(map(float, P_types_str.split(",")))

# Entrée des pièces
l_str = st.sidebar.text_input("Longueurs des pièces demandées (séparées par des virgules) [en mm]", "1000,1100")
l = list(map(float, l_str.split(",")))

# Tableau interactif des quantités demandées
st.subheader("📋 Quantités demandées pour chaque longueur de pièce")
df_input = pd.DataFrame({
    "Longueur demandée (mm)": l,
    "Quantité demandée": [1 for _ in l]
})
df_result = st.data_editor(df_input, num_rows="fixed")
L = df_result["Quantité demandée"].tolist()

if st.button("Optimiser"):
    X_used, bar_types_used, counts_per_type, gaspillage, total_matiere, total_waste = optimisation_min_matiere_cg(P_types, l, L)

    if X_used is None or X_used.shape[0] == 0:
        st.error("⚠️ Aucun plan optimal trouvé.")
    else:
        st.success(f"✅ Solution trouvée avec {len(bar_types_used)} barres utilisées !")

        # Plan de coupe
        df_plan = pd.DataFrame(
            X_used,
            columns=[f"Pièce {li}mm" for li in l],
            index=[f"Barre {i+1} ({int(bar_types_used[i])} m)" for i in range(len(bar_types_used))]
        )
        st.subheader("📊 Plan de coupe optimal")
        st.dataframe(df_plan)

        # Gaspillage par barre
        df_gaspillage = pd.DataFrame({
            "Barre": [f"Barre {i+1}" for i in range(len(bar_types_used))],
            "Longueur de barre [m]": bar_types_used,
            "Gaspillage [m]": gaspillage
        })
        st.subheader("🗑️ Gaspillage par barre")
        st.table(df_gaspillage)

        # Résumé global
        st.subheader("📈 Résumé global")
        st.write("Nombre de barres utilisées par type :")
        for ptype, cnt in counts_per_type.items():
            st.write(f"- {cnt} barre(s) de {ptype} m")

        st.metric("📏 Longueur totale de matière utilisée", f"{total_matiere} m")
        st.metric("🗑️ Gaspillage total", f"{total_waste} m")
