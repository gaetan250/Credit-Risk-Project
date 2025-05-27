import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from itertools import combinations

# Classe custom si existante
try:
    from EvaluatorClass import WLSModelEvaluator
except ImportError:
    WLSModelEvaluator = None  # Placeholder si non présent

# === Chargement des données ===
def load_segment_and_global_data(segment_path, global_path):
    df_segment = pd.read_csv(segment_path, sep=";")
    df_global = pd.read_csv(global_path, sep=";")
    return df_segment, df_global

def load_macro_data(macro_path):
    return pd.read_excel(macro_path)

# === Statistiques descriptives ===
def describe_dataframe(df, name="DataFrame"):
    print(f"=== Description de {name} ===")
    print(df.info())
    print(df.head())

# === Nettoyage des données ===
def clean_numeric_columns(df, columns):
    for col in columns:
        df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
    return df

def convert_global_dates(df):
    df['date'] = pd.to_datetime(df['date'])
    return df

def convert_segment_dates(df, source_col='cod_prd_ref'):
    df['date'] = pd.to_datetime(df[source_col].astype(str).str.strip().str.replace('T', 'Q'))
    return df

# === Visualisations ===
def plot_ccf_global_and_segment(df_global, df_segment):
    sns.set(style="whitegrid")
    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

    # Global
    sns.lineplot(data=df_global, x='date', y='Indicateur_Moyen_1_5', ax=axs[0], label='1_5')
    sns.lineplot(data=df_global, x='date', y='Indicateur_Moyen_1_6', ax=axs[0], label='1_6')
    axs[0].set_title("Évolution des Indicateurs Moyens 1_5 et 1_6 (Global)")

    # Segments
    unique_segments = sorted(df_segment['note_ref'].unique())
    for segment in unique_segments:
        df_plot_seg = df_segment[df_segment['note_ref'] == segment]
        sns.lineplot(
            data=df_plot_seg,
            x='date',
            y='Indicateur_moyen_Brut',
            ax=axs[1],
            label=f"Segment {segment}"
        )

    axs[1].set_title("Évolution de l'Indicateur Moyen Brut par Segment")
    axs[1].set_xlabel("Date")
    axs[1].set_ylabel("Indicateur Moyen Brut")
    axs[1].legend(title="Segment", loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()

# === Visualisation des données macroéconomiques ===
def plot_macro_variables(df_macro):
    df_macro = df_macro.copy()
    df_macro["date_dernier_mois"] = pd.to_datetime(df_macro["date_dernier_mois"])

    fig, ax = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    # PIB
    ax[0].plot(df_macro["date_dernier_mois"], df_macro["PIB"], label="PIB")
    ax[0].plot(df_macro["date_dernier_mois"], df_macro["PIB_diff1"], label="Variation PIB", linestyle='--')
    ax[0].set_title("Évolution du PIB")
    ax[0].legend()
    ax[0].grid(True)

    # IPL
    ax[1].plot(df_macro["date_dernier_mois"], df_macro["IPL"], label="IPL")
    ax[1].plot(df_macro["date_dernier_mois"], df_macro["IPL_diff1"], label="Variation IPL", linestyle='--')
    ax[1].set_title("Évolution de l’Indice des Prix des Logements")
    ax[1].legend()
    ax[1].grid(True)

    # Inflation
    ax[2].plot(df_macro["date_dernier_mois"], df_macro["Inflation"], label="Inflation")
    ax[2].plot(df_macro["date_dernier_mois"], df_macro["Inflation_diff1"], label="Variation Inflation", linestyle='--')
    ax[2].set_title("Évolution de l’Inflation")
    ax[2].legend()
    ax[2].grid(True)

    # Taux de chômage
    ax[3].plot(df_macro["date_dernier_mois"], df_macro["TCH"], label="Taux de Chômage")
    ax[3].plot(df_macro["date_dernier_mois"], df_macro["TCH_diff1"], label="Variation TCH", linestyle='--')
    ax[3].set_title("Évolution du Taux de Chômage et sa Variation")
    ax[3].legend()
    ax[3].grid(True)

    plt.xlabel("Date")
    plt.tight_layout()
    plt.show()

def plot_macro_correlation(df_macro):
    corr = df_macro[["PIB", "IPL", "Inflation", "TCH"]].corr()

    plt.figure(figsize=(4, 3))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Corrélation entre variables macroéconomiques")
    plt.tight_layout()
    plt.show()
# === Conversion de fin → début de trimestre ===
def start_of_quarter(date):
    if date.month == 3:
        return pd.Timestamp(year=date.year, month=1, day=1)
    elif date.month == 6:
        return pd.Timestamp(year=date.year, month=4, day=1)
    elif date.month == 9:
        return pd.Timestamp(year=date.year, month=7, day=1)
    elif date.month == 12:
        return pd.Timestamp(year=date.year, month=10, day=1)
    else:
        return pd.NaT

def standardize_macro_dates(df_macro):
    df_macro = df_macro.copy()
    df_macro["date_dernier_mois"] = pd.to_datetime(df_macro["date_dernier_mois"], errors='coerce')
    df_macro["date_dernier_mois"] = df_macro["date_dernier_mois"].apply(start_of_quarter)
    df_macro["date_dernier_mois"] = df_macro["date_dernier_mois"].dt.strftime("%d-%m-%Y")
    df_macro["date_dernier_mois"] = pd.to_datetime(df_macro["date_dernier_mois"], format="%d-%m-%Y")
    return df_macro

# === Comparaison des dates de 2009 entre jeux de données ===
def compare_dates_2009(df_global, df_segment, df_macro):
    dates_segment = df_segment[df_segment["date"].dt.year == 2009]["date"].unique()
    dates_global = df_global[df_global["date"].dt.year == 2009]["date"].unique()
    dates_macro = df_macro[df_macro["date_dernier_mois"].dt.year == 2009]["date_dernier_mois"].unique()

    df_dates_2009 = pd.DataFrame({
        "Dates df_global (2009)": dates_global,
        "Dates df_segment (2009)": dates_segment,
        "Dates df_macro (2009)": dates_macro
    })

    print(df_dates_2009)
    return df_dates_2009

# === Fusion des données segment + macroéconomie avec lags ===
def create_merged_dataset(df_segment, df_macro):
    # Création d'une copie pour lags
    df_macro_lag = df_macro.copy()

    df_macro_lag['PIB_lag1'] = df_macro_lag['PIB'].shift(1)
    df_macro_lag['IPL_lag1'] = df_macro_lag['IPL_diff1'].shift(1)
    df_macro_lag['Inflation_lag1'] = df_macro_lag['Inflation_diff1'].shift(1)
    df_macro_lag['TCH_lag1'] = df_macro_lag['TCH_diff1'].shift(1)

    # Pivot des données de segment pour obtenir une colonne par note_ref
    pivot_df = df_segment.pivot(index="date", columns="note_ref", values=["Indicateur_moyen_Brut", "PourcNoteCohorte5"])
    pivot_df.columns = [f"{var}_{int(note)}" for var, note in pivot_df.columns]

    # Fusion des macro avec pivot_df
    df_macro_lag = df_macro_lag.rename(columns={"date_dernier_mois": "date"})
    df_merged = pivot_df.merge(df_macro_lag, on="date", how="left")

    # Version sans la colonne Indicateur_moyen_Brut_6
    df_merged_6 = df_merged.drop(columns=["Indicateur_moyen_Brut_6"], inplace=False)

    # Définir l'index sur la date
    df_merged.set_index("date", inplace=True)
    df_merged_6.set_index("date", inplace=True)

    return df_merged, df_merged_6
# === Tracer les séries temporelles de df_merged ===
def plot_merged_series(df_merged, title="Séries temporelles de df_merged"):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(18, 8))

    # Supprimer les colonnes _diff1 ou _lag1
    df_base = df_merged.loc[:, ~df_merged.columns.str.endswith(('_diff1', '_lag1'))]

    # Identifier les 4 dernières colonnes restantes
    last_four_columns = df_base.columns[-4:]

    for column in df_base.columns:
        if column in last_four_columns:
            plt.plot(df_base.index, df_base[column], label=column, linestyle='--')  # pointillés
        else:
            plt.plot(df_base.index, df_base[column], label=column)  # continu

    plt.title(title, fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Valeur", fontsize=12)
    plt.ylim([-3, 3])
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=10)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


# === Évolution des indicateurs globaux et par segments (copie si non encore importée) ===
def plot_global_and_segment_evolution(df_global, df_segment):
    sns.set(style="whitegrid")
    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

    # Global
    sns.lineplot(data=df_global, x='date', y='Indicateur_Moyen_1_5', ax=axs[0], label='1_5')
    sns.lineplot(data=df_global, x='date', y='Indicateur_Moyen_1_6', ax=axs[0], label='1_6')
    axs[0].set_title("Évolution des Indicateurs Moyens 1_5 et 1_6 (Global)")

    # Segments
    unique_segments = sorted(df_segment['note_ref'].unique())
    for segment in unique_segments:
        df_plot_seg = df_segment[df_segment['note_ref'] == segment]
        sns.lineplot(
            data=df_plot_seg,
            x='date',
            y='Indicateur_moyen_Brut',
            ax=axs[1],
            label=f"Segment {segment}"
        )

    axs[1].set_title("Évolution de l'Indicateur Moyen Brut par Segment")
    axs[1].set_xlabel("Date")
    axs[1].set_ylabel("Indicateur Moyen Brut")
    axs[1].legend(title="Segment", loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.show()

# === Visualisation : PourcNote vs Indicateur moyen brut par segment ===
def plot_pourc_note_vs_indicateur(df_merged):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12, 18), sharex=True)

    for i in range(1, 6):
        ax = axes[i - 1]
        try:
            ax.plot(df_merged.index, df_merged[f'Indicateur_moyen_Brut_{i}'], label=f'Indicateur_moyen_Brut_{i}')
            ax.plot(df_merged.index, df_merged[f'PourcNoteCohorte5_{i}'], label=f'PourcNoteCohorte5_{i}')
            ax.set_title(f'Segment {i}')
            ax.legend()
            ax.grid(True)
        except KeyError as e:
            print(f"Colonne manquante pour segment {i}: {e}")

    plt.tight_layout()
    plt.show()


# === Visualisation : Autocorrélations ===
def plot_autocorrelations(df_global, df_segment):
    import matplotlib.pyplot as plt
    from pandas.plotting import autocorrelation_plot

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    autocorrelation_plot(df_global["Indicateur_Moyen_1_5"].dropna())
    plt.title("Autocorrélation - Indicateur_Moyen_1_5")

    plt.subplot(2, 2, 2)
    autocorrelation_plot(df_global["Indicateur_Moyen_1_6"].dropna())
    plt.title("Autocorrélation - Indicateur_Moyen_1_6")

    plt.subplot(2, 2, 3)
    autocorrelation_plot(df_segment["Indicateur_moyen_Brut"].dropna())
    plt.title("Autocorrélation - Indicateur_Moyen_Brut (Segment)")

    plt.tight_layout()
    plt.show()

from statsmodels.tsa.stattools import adfuller

# === Test ADF global sur des séries nommées ===
def run_adf_tests_named(series_dict):
    results = []
    for name, series in series_dict.items():
        series = series.dropna()
        try:
            adf_result = adfuller(series)
            results.append({
                "Série": name,
                "ADF Statistic": adf_result[0],
                "p-value": adf_result[1],
                "Critique 1%": adf_result[4]['1%'],
                "Critique 5%": adf_result[4]['5%'],
                "Critique 10%": adf_result[4]['10%']
            })
        except Exception as e:
            results.append({
                "Série": name,
                "ADF Statistic": None,
                "p-value": None,
                "Critique 1%": None,
                "Critique 5%": None,
                "Critique 10%": None,
                "Erreur": str(e)
            })
    return pd.DataFrame(results)

# === Test ADF par segment (note_ref) ===
def run_adf_tests_by_segment(df_segment):
    results = []
    for note, group in df_segment.groupby("note_ref"):
        series = group.sort_values("date")["Indicateur_moyen_Brut"]

        if series.nunique() == 1:
            results.append({
                'Série': f'Indicateur_Moyen_Brut_{note}',
                'ADF Statistic': None,
                'p-value': None,
                'Critique 1%': None,
                'Critique 5%': None,
                'Critique 10%': None,
                'Note': 'Constant series - skipped'
            })
            continue

        adf = adfuller(series.dropna())
        results.append({
            'Série': f'Indicateur_Moyen_Brut_{note}',
            'ADF Statistic': adf[0],
            'p-value': adf[1],
            'Critique 1%': adf[4]['1%'],
            'Critique 5%': adf[4]['5%'],
            'Critique 10%': adf[4]['10%']
        })
    return pd.DataFrame(results)

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# === Filtrer les données macro à partir de 2009
def filter_macro_since_2009(df_macro):
    return df_macro[df_macro['date_dernier_mois'] >= '2009-01-01'].copy()


# === ADF + ACF/PACF sur variables macro ===
def analyze_macro_stationarity(df_macro, macro_vars=["PIB", "IPL", "Inflation", "TCH"], lags=20):
    import matplotlib.pyplot as plt

    adf_results_macro = []
    fig, axes = plt.subplots(nrows=len(macro_vars), ncols=2, figsize=(14, 4 * len(macro_vars)))

    for i, var in enumerate(macro_vars):
        series = df_macro[var].dropna()

        # Tracer ACF
        plot_acf(series, ax=axes[i][0], lags=lags, alpha=0.05)
        axes[i][0].set_title(f"ACF - {var}")

        # Tracer PACF
        plot_pacf(series, ax=axes[i][1], lags=lags, alpha=0.05, method="ywmle")
        axes[i][1].set_title(f"PACF - {var}")

        # Test ADF
        try:
            adf = adfuller(series)
            adf_results_macro.append({
                "Variable": var,
                "ADF Statistic": adf[0],
                "p-value": adf[1],
                "Critique 1%": adf[4]["1%"],
                "Critique 5%": adf[4]["5%"],
                "Critique 10%": adf[4]["10%"]
            })
        except Exception as e:
            adf_results_macro.append({
                "Variable": var,
                "ADF Statistic": None,
                "p-value": None,
                "Critique 1%": None,
                "Critique 5%": None,
                "Critique 10%": None,
                "Erreur": str(e)
            })

    plt.tight_layout()
    plt.show()

    return pd.DataFrame(adf_results_macro)

from statsmodels.tsa.filters.hp_filter import hpfilter

# === Test ADF sur les log-diff de séries ===
def run_adf_logdiff(series_dict):
    results = []
    for name, series in series_dict.items():
        log_diff = np.log(series).diff().dropna().replace([np.inf, -np.inf], np.nan).dropna()
        try:
            adf = adfuller(log_diff)
            results.append({
                "Série (log_diff)": name,
                "ADF Statistic": adf[0],
                "p-value": adf[1],
                "Critique 1%": adf[4]['1%'],
                "Critique 5%": adf[4]['5%'],
                "Critique 10%": adf[4]['10%']
            })
        except:
            results.append({
                "Série (log_diff)": name,
                "ADF Statistic": None,
                "p-value": None,
                "Critique 1%": None,
                "Critique 5%": None,
                "Critique 10%": None
            })
    return pd.DataFrame(results)

# === Appliquer transformations et tests ADF sur segments ===
def transform_and_test_segments(segments_dfs, methods=["logdiff", "pctchange", "hpfilter"]):
    from statsmodels.tsa.stattools import adfuller

    results = {
        "logdiff": {},
        "pctchange": {},
        "hpfilter": {},
        "adf_logdiff": [],
        "adf_pctchange": []
    }

    for name in segments_dfs:
        df = segments_dfs[name].copy()

        # log-diff
        df_logdiff = df[["date", "Indicateur_moyen_Brut"]].copy()
        df_logdiff["log_diff"] = np.log(df_logdiff["Indicateur_moyen_Brut"]).diff()
        results["logdiff"][name] = df_logdiff.dropna().reset_index(drop=True)

        # pct-change
        df_pct = df[["date", "Indicateur_moyen_Brut"]].copy()
        df_pct["pct_change"] = df_pct["Indicateur_moyen_Brut"].pct_change()
        results["pctchange"][name] = df_pct.dropna().reset_index(drop=True)

        # hp-filter
        cycle, trend = hpfilter(df["Indicateur_moyen_Brut"], lamb=1600)
        df_hp = df[["date"]].copy()
        df_hp["cycle"] = cycle
        results["hpfilter"][name] = df_hp.reset_index(drop=True)

        # ADF sur logdiff
        try:
            adf = adfuller(results["logdiff"][name]["log_diff"])
            results["adf_logdiff"].append({
                "Segment": name,
                "ADF Statistic": adf[0],
                "p-value": adf[1],
                "Critique 1%": adf[4]['1%'],
                "Critique 5%": adf[4]['5%'],
                "Critique 10%": adf[4]['10%']
            })
        except:
            results["adf_logdiff"].append({
                "Segment": name, "ADF Statistic": None, "p-value": None,
                "Critique 1%": None, "Critique 5%": None, "Critique 10%": None
            })

        # ADF sur pct_change
        try:
            adf = adfuller(results["pctchange"][name]["pct_change"])
            results["adf_pctchange"].append({
                "Segment": name,
                "ADF Statistic": adf[0],
                "p-value": adf[1],
                "Critique 1%": adf[4]['1%'],
                "Critique 5%": adf[4]['5%'],
                "Critique 10%": adf[4]['10%']
            })
        except:
            results["adf_pctchange"].append({
                "Segment": name, "ADF Statistic": None, "p-value": None,
                "Critique 1%": None, "Critique 5%": None, "Critique 10%": None
            })

    return results

# === Visualisation comparative des transformations ===
def plot_transformed_segments(segments_dfs, logdiff_dfs, pctchange_dfs, hp_cycle_dfs):
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True)

    for i, seg in enumerate(["segment_2", "segment_3"]):
        row = 0 if seg == "segment_2" else 1

        # Original vs log(diff)
        axes[row, 0].plot(segments_dfs[seg]["date"], segments_dfs[seg]["Indicateur_moyen_Brut"], color="gray", label="Original")
        axes[row, 0].plot(logdiff_dfs[seg]["date"], logdiff_dfs[seg]["log_diff"], color="green" if row == 0 else "red", label="log(diff)")
        axes[row, 0].set_title(f"{seg} - log(diff)")
        axes[row, 0].legend()

        # Original vs pct_change
        axes[row, 1].plot(segments_dfs[seg]["date"], segments_dfs[seg]["Indicateur_moyen_Brut"], color="gray", label="Original")
        axes[row, 1].plot(pctchange_dfs[seg]["date"], pctchange_dfs[seg]["pct_change"], color="green" if row == 0 else "red", label="pct_change")
        axes[row, 1].set_title(f"{seg} - Taux de variation (%)")
        axes[row, 1].legend()

        # Original vs HP cycle
        axes[row, 2].plot(segments_dfs[seg]["date"], segments_dfs[seg]["Indicateur_moyen_Brut"], color="gray", label="Original")
        axes[row, 2].plot(hp_cycle_dfs[seg]["date"], hp_cycle_dfs[seg]["cycle"], color="green" if row == 0 else "red", label="Cycle (HP)")
        axes[row, 2].set_title(f"{seg} - Cycle (HP filter)")
        axes[row, 2].legend()

    plt.tight_layout()
    plt.show()
# === Refaire le test ADF sur les séries finales et tracer ACF ===
def recheck_stationarity(series_dict, lags=20):
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf
    import matplotlib.pyplot as plt

    adf_results = []
    fig, axes = plt.subplots(1, len(series_dict), figsize=(7 * len(series_dict), 3))

    if len(series_dict) == 1:
        axes = [axes]

    for i, (name, serie) in enumerate(series_dict.items()):
        serie = serie.dropna()
        try:
            adf = adfuller(serie)
            adf_results.append({
                "Segment": name,
                "ADF Statistic": adf[0],
                "p-value": adf[1],
                "Critique 1%": adf[4]["1%"],
                "Critique 5%": adf[4]["5%"],
                "Critique 10%": adf[4]["10%"]
            })
        except:
            adf_results.append({
                "Segment": name,
                "ADF Statistic": None,
                "p-value": None,
                "Critique 1%": None,
                "Critique 5%": None,
                "Critique 10%": None
            })

        plot_acf(serie, ax=axes[i], lags=lags, alpha=0.05)
        axes[i].set_title(f"ACF - {name}")

    plt.tight_layout()
    plt.show()
    return pd.DataFrame(adf_results)

# === Injecter les séries transformées dans df_merged ===
def inject_transformed_segments(df_merged, segments_dfs, logdiff_dfs, hp_cycle_dfs):
    df_merged = df_merged.iloc[1:].reset_index(drop=True)

    df_merged["Indicateur_moyen_Brut_2_diff"] = hp_cycle_dfs["segment_2"]["cycle"].reset_index(drop=True)
    df_merged["Indicateur_moyen_Brut_3_diff"] = logdiff_dfs["segment_3"]["log_diff"].reset_index(drop=True)

    # Réattribuer les dates depuis segment_1
    df_merged["date"] = segments_dfs["segment_1"]["date"].iloc[1:].reset_index(drop=True)
    df_merged.set_index("date", inplace=True)
    return df_merged

# === Visualiser les séries stationnarisées finales ===
def plot_final_stationary_series(df_merged, segment_cols=None):
    if segment_cols is None:
        segment_cols = [
            'Indicateur_moyen_Brut_1',
            'Indicateur_moyen_Brut_2_diff',
            'Indicateur_moyen_Brut_3_diff',
            'Indicateur_moyen_Brut_4',
            'Indicateur_moyen_Brut_5'
        ]

    plt.figure(figsize=(14, 6))
    for name in segment_cols:
        if name in df_merged.columns:
            plt.plot(df_merged.index, df_merged[name], label=name)

    plt.title("Séries finales stationnarisées (segments 1 à 6)")
    plt.xlabel("Date")
    plt.ylabel("Valeur")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.legend()
    plt.tight_layout()
    plt.show()

# === Nettoyage de colonnes spécifiques ===
def drop_unused_columns(df, columns_to_drop=["Indicateur_moyen_Brut_6", "PourcNoteCohorte5_6"]):
    return df.drop(columns=columns_to_drop, errors="ignore")

# === Heatmap complète de corrélation ===
def plot_full_correlation_matrix(df):
    corr = df.corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Corrélation entre toutes les variables")
    plt.tight_layout()
    plt.show()

# === Corrélations ciblées : Indicateurs vs Macro (niveau et variations) ===
def plot_correlation_with_macro(df):
    cols_indicateur = [col for col in df.columns if col.startswith("Indicateur")]
    cols_macro = ["PIB", "IPL", "TCH", "Inflation"]
    cols_macro_diff = ["PIB_diff1", "IPL_diff1", "TCH_diff1", "Inflation_diff1"]

    # Corrélations
    corr_macro = df[cols_indicateur + cols_macro].corr().loc[cols_indicateur, cols_macro]
    corr_macro_diff = df[cols_indicateur + cols_macro_diff].corr().loc[cols_indicateur, cols_macro_diff]

    # Tracé
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    sns.heatmap(corr_macro, annot=True, fmt=".2f", cmap="coolwarm", square=True, ax=axes[0])
    axes[0].set_title("Corrélation Indicateurs ↔ Macro (niveau)")

    sns.heatmap(corr_macro_diff, annot=True, fmt=".2f", cmap="coolwarm", square=True, ax=axes[1])
    axes[1].set_title("Corrélation Indicateurs ↔ Macro (diff1)")

    plt.tight_layout()
    plt.show()

    return corr_macro, corr_macro_diff
