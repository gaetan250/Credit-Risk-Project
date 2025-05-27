import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm


def test_coint_engle_granger(df, vars_i1):
    results_eg = []
    for var1, var2 in combinations(vars_i1, 2):
        series1 = df[var1].dropna()
        series2 = df[var2].dropna()
        common_index = series1.index.intersection(series2.index)
        stat, pval, _ = coint(series1.loc[common_index], series2.loc[common_index])
        results_eg.append({
            "Série 1": var1,
            "Série 2": var2,
            "Stat cointégration": stat,
            "p-value": pval,
            "Cointégré": pval < 0.05
        })
    return pd.DataFrame(results_eg).sort_values("p-value")


def test_coint_johansen(df, vars_i1, group_size=3):
    results_johansen = []
    for group in combinations(vars_i1, group_size):
        data = df[list(group)].dropna()
        if len(data) < 30:
            continue
        joh = coint_johansen(data, det_order=0, k_ar_diff=1)
        trace_stats = joh.lr1
        cvt_95 = joh.cvt[:, 1]
        r = sum(trace_stats > cvt_95)
        results_johansen.append({
            "Variables": group,
            "Nombre relations cointégrées (5%)": r
        })
    return pd.DataFrame(results_johansen)


def extract_stationary_combination(df, variables):
    data = df[variables].dropna()
    joh = coint_johansen(data, det_order=0, k_ar_diff=1)
    beta = joh.evec[:, 0]
    z_coint = data.values @ beta
    pval_adf = adfuller(z_coint)[1]
    return beta, z_coint, pval_adf
# ==================== MODELE WLS ====================

class WLSModelEvaluator:
    def __init__(self, df, macro_vars, target_vars):
        self.df = df
        self.macro_vars = macro_vars
        self.target_vars = target_vars
        self.results = {}

    def fit_models(self):
        for target in self.target_vars:
            data = self.df[self.macro_vars + [target]].dropna()
            X = sm.add_constant(data[self.macro_vars])
            y = data[target]
            weights = data[target].abs()
            model = sm.WLS(y, X, weights=weights)
            res = model.fit()
            self.results[target] = res

    def display_results(self):
        for target, res in self.results.items():
            print("===========================")
            print(f"Modèle pour : {target}")
            print(res.summary())

    def plot_diagnostics(self):
        import seaborn as sns
        import scipy.stats as stats
        from statsmodels.graphics.tsaplots import plot_acf

        n = len(self.target_vars)
        fig, axes = plt.subplots(4, n, figsize=(5 * n, 16))
        fig.suptitle("Diagnostics des Résidus", fontsize=18)

        for i, target in enumerate(self.target_vars):
            res = self.results[target]
            resid = res.resid
            fitted = res.fittedvalues

            # Ligne 1: résidus vs valeurs prédites
            sns.scatterplot(x=fitted, y=resid, ax=axes[0, i])
            axes[0, i].axhline(0, color="red", linestyle="--")
            axes[0, i].set_title(f"{target}")
            axes[0, i].set_xlabel("Prédictions")
            axes[0, i].set_ylabel("Résidus")

            # Ligne 2: histogramme des résidus
            sns.histplot(resid, kde=True, ax=axes[1, i], bins=15)
            axes[1, i].set_title(f"{target}")
            axes[1, i].set_xlabel("Résidus")
            axes[1, i].set_ylabel("Count")

            # Ligne 3: QQ-plot
            stats.probplot(resid, dist="norm", plot=axes[2, i])
            axes[2, i].set_title(f"{target}")

            # Ligne 4: ACF
            plot_acf(resid, ax=axes[3, i], title=f"{target}", zero=False)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


def create_macro_features(df):
    df = df.copy()

    # Dates en datetime
    df.index = pd.to_datetime(df.index)
    df['year'] = df.index.year
    df['quarter'] = df.index.quarter

    # Transformations de base
    df['PIB_squared'] = df['PIB'] ** 2
    df['log_TCH_diff1'] = np.log1p(df['TCH_diff1'].abs())
    df['zscore_PIB'] = (df['PIB'] - df['PIB'].mean()) / df['PIB'].std()

    # Interactions classiques
    df['PIBxIPL'] = df['PIB'] * df['IPL_diff1']
    df['PIBxInflation'] = df['PIB'] * df['Inflation_diff1']
    df['PIBxTCH'] = df['PIB'] * df['TCH_diff1']
    df['IPLxInflation'] = df['IPL_diff1'] * df['Inflation_diff1']
    df['IPLxTCH'] = df['IPL_diff1'] * df['TCH_diff1']
    df['InflationxTCH'] = df['Inflation_diff1'] * df['TCH_diff1']

    # Décalages
    df['PIB_lag2'] = df['PIB_lag1'].shift(1)

    # Moyennes glissantes
    df['TCH_ma3'] = df['TCH_diff1'].rolling(3).mean()
    df['Inflation_ma3'] = df['Inflation_diff1'].rolling(3).mean()
    df['PIB_ma6'] = df['PIB'].rolling(6).mean()

    # Volatilité
    df['Inflation_std3'] = df['Inflation_diff1'].rolling(3).std()
    df['TCH_volatility'] = df['TCH_diff1'].rolling(5).std()

    # Accélérations
    df['PIB_acc'] = df['PIB'].diff().diff()
    df['Inflation_acc'] = df['Inflation_diff1'].diff()
    df['TCH_acc'] = df['TCH_diff1'].diff()

    # Ratios
    df['PIB_to_IPL'] = df['PIB'] / (df['IPL_diff1'].replace(0, np.nan) + 1e-6)
    df['TCH_to_Inflation'] = df['TCH_diff1'] / (df['Inflation_diff1'].replace(0, np.nan) + 1e-6)
    df['IPL_to_TCH'] = df['IPL_diff1'] / (df['TCH_diff1'].replace(0, np.nan) + 1e-6)

    # Écarts à la moyenne
    df['PIB_gap_ma3'] = df['PIB'] - df['PIB'].rolling(3).mean()

    # Croissances glissantes
    df['PIB_growth_3q'] = df['PIB'].pct_change(periods=3)
    df['Inflation_growth_3q'] = df['Inflation_diff1'].pct_change(periods=3)

    # Indicateurs binaires
    df['recession_flag'] = (df['PIB'] < 0).astype(int)
    df['crisis_flag'] = ((df.index >= '2008-09') & (df.index <= '2009-12')).astype(int)
    df['pre_covid'] = (df.index < '2020-03').astype(int)

    # Composantes cycliques
    df['PIB_cycle'], df['PIB_trend'] = hpfilter(df['PIB'], lamb=1600)

    # Fréquences (FFT)
    freqs = np.fft.fft(df['PIB'].fillna(0))
    df['PIB_freq1'] = np.real(freqs[1])

    df['Inflation_surge'] = (
        df['Inflation_diff1'] > df['Inflation_diff1'].rolling(4).mean() + 2 * df['Inflation_diff1'].rolling(4).std()
    ).astype(int)

    # Recalcul TCH_ma3 et TCH_volatility au cas où ils sont écrasés
    df['TCH_ma3'] = df['TCH_diff1'].rolling(3).mean()
    df['TCH_volatility'] = df['TCH_diff1'].rolling(5).std()

    return df


def compute_vif(df, macro_vars):
    df_filtered = df.dropna(subset=macro_vars).copy()
    X = df_filtered[macro_vars]
    X_const = add_constant(X)
    vif_data = pd.DataFrame()
    vif_data['variable'] = X_const.columns
    vif_data['VIF'] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
    return vif_data.sort_values(by='VIF', ascending=False)
    return df

# model.py – Partie PLS et évaluation de la normalité des résidus

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import r2_score
from scipy.stats import shapiro, jarque_bera
import statsmodels.api as sm


def compute_pls_components(df, macro_vars):
    df = df.copy()
    for i in range(1, 6):
        pls = PLSRegression(n_components=2)
        X = df[macro_vars]
        y = df[f'Indicateur_moyen_Brut_{i}']
        pls.fit(X, y)
        df['PLS1'] = pls.transform(X)[:, 0]
        df['PLS2'] = pls.transform(X)[:, 1]

        y_pred_pls = pls.predict(X)
        ss_residual = np.sum((y - y_pred_pls.ravel()) ** 2)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_residual / ss_total)

        print(f"R² du modèle PLS : {r2:.4f}")

    return df


def evaluate_residual_normality(df, macro_vars, target_col="Indicateur_moyen_Brut_5"):
    X = df[macro_vars].copy()
    y = df[target_col]

    results = []

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X, y)
    y_pred_rf = rf.predict(X)
    res_rf = y - y_pred_rf
    r2_rf = r2_score(y, y_pred_rf)
    shapiro_rf = shapiro(res_rf)
    jb_rf = jarque_bera(res_rf)
    results.append({
        "Modèle": "Random Forest",
        "R²": round(r2_rf, 4),
        "Shapiro p-val": round(shapiro_rf.pvalue, 4),
        "Shapiro": "Normaux" if shapiro_rf.pvalue > 0.05 else "Non normaux",
        "Jarque-Bera p-val": round(jb_rf.pvalue, 4),
        "Jarque-Bera": "Normaux" if jb_rf.pvalue > 0.05 else "Non normaux"
    })

    # Huber Regressor
    huber = HuberRegressor()
    huber.fit(X, y)
    y_pred_huber = huber.predict(X)
    res_huber = y - y_pred_huber
    r2_huber = r2_score(y, y_pred_huber)
    shapiro_h = shapiro(res_huber)
    jb_h = jarque_bera(res_huber)
    results.append({
        "Modèle": "Huber Regressor",
        "R²": round(r2_huber, 4),
        "Shapiro p-val": round(shapiro_h.pvalue, 4),
        "Shapiro": "Normaux" if shapiro_h.pvalue > 0.05 else "Non normaux",
        "Jarque-Bera p-val": round(jb_h.pvalue, 4),
        "Jarque-Bera": "Normaux" if jb_h.pvalue > 0.05 else "Non normaux"
    })

    # Quantile Regression
    mod = sm.QuantReg(y, sm.add_constant(X))
    res = mod.fit(q=0.5)
    y_pred_qr = res.predict(sm.add_constant(X))
    res_qr = y - y_pred_qr
    r2_qr = r2_score(y, y_pred_qr)
    shapiro_qr = shapiro(res_qr)
    jb_qr = jarque_bera(res_qr)
    results.append({
        "Modèle": "Quantile Regression (q=0.5)",
        "R²": round(r2_qr, 4),
        "Shapiro p-val": round(shapiro_qr.pvalue, 4),
        "Shapiro": "Normaux" if shapiro_qr.pvalue > 0.05 else "Non normaux",
        "Jarque-Bera p-val": round(jb_qr.pvalue, 4),
        "Jarque-Bera": "Normaux" if jb_qr.pvalue > 0.05 else "Non normaux"
    })

    return pd.DataFrame(results)


def select_best_linear_models(df, macro_vars, segments):
    results_model = []
    results_modelW = []

    for seg in segments:
        X = df[macro_vars].copy()
        X = sm.add_constant(X)
        try:
            y = df[seg + '_diff']
        except:
            y = df[seg]

        segment_num = seg.split('_')[3].split('_')[0]
        weights = df[f'PourcNoteCohorte5_{segment_num}']

        best_r2 = -np.inf
        best_model = None
        best_k = None

        best_r2w = -np.inf
        best_modelW = None
        best_kw = None

        for k in range(2, len(macro_vars)):
            corrs = X.drop(columns='const').apply(lambda col: np.corrcoef(col, y)[0, 1])
            X_top_cols = corrs.abs().sort_values(ascending=False).head(k).index.tolist()
            X_top_corr = sm.add_constant(X[X_top_cols], has_constant='add')

            try:
                model = sm.OLS(y, X_top_corr).fit()
                modelW = sm.WLS(y, X_top_corr, weights=weights).fit()
            except:
                continue

            if model.rsquared > best_r2:
                best_r2 = model.rsquared
                best_model = model
                best_k = k

            if modelW.rsquared > best_r2w:
                best_r2w = modelW.rsquared
                best_modelW = modelW
                best_kw = k

        results_model.append({
            "Segment": seg,
            "Model": "OLS",
            "k": best_k,
            "R2_score": round(best_r2, 4),
            "Coefficients": best_model.params.to_dict()
        })

        results_modelW.append({
            "Segment": seg,
            "Model": "WLS",
            "k": best_kw,
            "R2_score": round(best_r2w, 4),
            "Coefficients": best_modelW.params.to_dict()
        })

    df_ols = pd.DataFrame(results_model)
    df_wls = pd.DataFrame(results_modelW)
    return df_ols, df_wls
