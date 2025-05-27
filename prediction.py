# prediction.py – Complet avec projections linéaires, Gradient Boosting, visualisations et forward-looking

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

# === Dernières valeurs observées ===
def get_last_real_values():
    return {"PIB": 0.868858, "IPL": -0.455149, "TCH": 7.3, "Inflation": 3.705326}


# === Projection linéaire à partir de coefficients ===
def generate_projection_df(results_df, scenarios_by_quarter, last_real=None):
    if last_real is None:
        last_real = get_last_real_values()

    projection_rows = []

    for _, row in results_df.iterrows():
        segment = row["Segment"]
        coefs = row["Coefficients"]

        for trimestre, scenarios in scenarios_by_quarter.items():
            for scenario, year_data in scenarios.items():
                prev = last_real.copy()
                for year, values in year_data.items():
                    features = {
                        "PIB_diff1": values["PIB"] - prev["PIB"],
                        "IPL_diff1": values["IPL"] - prev["IPL"],
                        "TCH_diff1": values["TCH"] - prev["TCH"],
                        "Inflation_diff1": values["Inflation"] - prev["Inflation"],
                        "PIB_lag1": prev["PIB"],
                        "IPL_lag1": prev["IPL"],
                        "TCH_lag1": prev["TCH"],
                        "Inflation_lag1": prev["Inflation"]
                    }

                    y_pred = coefs.get("const", 0) + sum(
                        coefs.get(k, 0) * v for k, v in features.items()
                    )

                    projection_rows.append({
                        "Segment": segment,
                        "Scénario": scenario,
                        "Année": year,
                        "Trimestre": trimestre,
                        "Prévision_CCF": round(y_pred, 4)
                    })

                    prev = values

    projection_df = pd.DataFrame(projection_rows)
    projection_df["Période"] = projection_df["Année"].astype(str) + "-" + projection_df["Trimestre"]
    return projection_df


# === Projection avec Gradient Boosting ===
def train_gradient_boosting_models(df, macro_vars, segments):
    gb_models = {}
    for target in segments:
        data = df.dropna(subset=macro_vars + [target])
        X = data[[
            "PIB_diff1", "IPL_diff1", "TCH_diff1", "Inflation_diff1",
            "PIB_lag1", "IPL_lag1", "TCH_lag1", "Inflation_lag1"
        ]]
        y = data[target]
        model = GradientBoostingRegressor().fit(X, y)
        gb_models[target] = model
    return gb_models


def generate_projection_gb(gb_models, scenarios_by_quarter, last_real=None):
    if last_real is None:
        last_real = get_last_real_values()

    projection_rows = []

    for segment, model in gb_models.items():
        for trimestre, scenarios in scenarios_by_quarter.items():
            for scenario, year_data in scenarios.items():
                prev = last_real.copy()
                for year, values in year_data.items():
                    X_macro = pd.DataFrame([{
                        "PIB_diff1": values["PIB"] - prev["PIB"],
                        "IPL_diff1": values["IPL"] - prev["IPL"],
                        "TCH_diff1": values["TCH"] - prev["TCH"],
                        "Inflation_diff1": values["Inflation"] - prev["Inflation"],
                        "PIB_lag1": prev["PIB"],
                        "IPL_lag1": prev["IPL"],
                        "TCH_lag1": prev["TCH"],
                        "Inflation_lag1": prev["Inflation"]
                    }])

                    y_pred = model.predict(X_macro)[0]

                    projection_rows.append({
                        "Segment": segment,
                        "Scénario": scenario,
                        "Année": year,
                        "Trimestre": trimestre,
                        "Prévision_CCF": round(y_pred, 4)
                    })

                    prev = values

    projection_df = pd.DataFrame(projection_rows)
    projection_df["Période"] = projection_df["Année"].astype(str) + "-" + projection_df["Trimestre"]
    return projection_df


# === Moyenne pondérée des scénarios ===
def compute_forward_looking(projection_df, poids_scenarios=None):
    if poids_scenarios is None:
        poids_scenarios = {"PESS": 0.25, "CENT": 0.50, "OPTI": 0.25}

    projection_df = projection_df.copy()
    projection_df["Poids"] = projection_df["Scénario"].map(poids_scenarios)
    df_fl = (
        projection_df
        .assign(pondéré=projection_df["Prévision_CCF"] * projection_df["Poids"])
        .groupby(["Segment", "Année", "Trimestre", "Période"], as_index=False)
        .agg({"pondéré": "sum"})
        .rename(columns={"pondéré": "CCF_Forward_Looking"})
    )
    return df_fl


# === Visualisation combinée historique + projections ===
def plot_forecast_with_history(projection_df, df_real_melted):
    projection_df["Période"] = projection_df["Année"].astype(str) + "-" + projection_df["Trimestre"]
    df_real_melted["Période"] = df_real_melted["Année"].astype(str) + "-" + df_real_melted["Trimestre"]

    ordre_periodes = sorted(set(projection_df["Période"]) | set(df_real_melted["Période"]))
    segments = projection_df["Segment"].unique()
    couleurs_scenarios = {"PESS": "red", "CENT": "orange", "OPTI": "green"}
    couleurs_segments = plt.cm.tab10.colors

    plt.figure(figsize=(18, 8))

    for i, segment in enumerate(segments):
        df_hist = df_real_melted[df_real_melted["Segment"] == segment]
        plt.plot(df_hist["Période"], df_hist["Valeur_réelle"], linestyle='--', marker='x',
                 label=f"{segment} - Historique", color=couleurs_segments[i % len(couleurs_segments)])

    for i, segment in enumerate(segments):
        df_pred = projection_df[projection_df["Segment"] == segment]
        for scenario, color in couleurs_scenarios.items():
            df_scen = df_pred[df_pred["Scénario"] == scenario]
            plt.plot(df_scen["Période"], df_scen["Prévision_CCF"],
                     marker='x', linewidth=1.5, alpha=0.7,
                     label=f"{segment} - {scenario}", color=color)

    plt.xticks(ordre_periodes, rotation=45, ha='right', fontsize=9)
    plt.xlabel("Période")
    plt.ylabel("CCF")
    plt.title("CCF Historique + Prévisions - Tous Segments et Scénarios")
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.tight_layout()
    plt.show()


# === Visualisation Forward Looking ===
def plot_forward_looking(df_fl, df_real_melted, projection_df):
    projection_df["Période"] = projection_df["Année"].astype(str) + "-" + projection_df["Trimestre"]
    df_real_melted["Période"] = df_real_melted["Année"].astype(str) + "-" + df_real_melted["Trimestre"]

    ordre_periodes = sorted(set(projection_df["Période"]) | set(df_real_melted["Période"]))
    segments = projection_df["Segment"].unique()
    couleurs_segments = plt.cm.tab10.colors

    plt.figure(figsize=(18, 8))

    for i, segment in enumerate(segments):
        df_hist = df_real_melted[df_real_melted["Segment"] == segment]
        plt.plot(df_hist["Période"], df_hist["Valeur_réelle"], linestyle='--', marker='x',
                 label=f"{segment} - Historique", color=couleurs_segments[i % len(couleurs_segments)])

    for i, segment in enumerate(segments):
        df_seg_fl = df_fl[df_fl["Segment"] == segment]
        plt.plot(df_seg_fl["Période"], df_seg_fl["CCF_Forward_Looking"], linestyle='--', marker='x',
                 label=f"{segment} - FL pondéré", color=couleurs_segments[i % len(couleurs_segments)])

    plt.xticks(ordre_periodes, rotation=45, ha='right', fontsize=9)
    plt.xlabel("Période")
    plt.ylabel("CCF")
    plt.title("CCF Historique, Scénarios et Projection Forward-Looking")
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
    plt.tight_layout()
    plt.show()
