import pandas as pd
import numpy as np
from itertools import combinations
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.regression.linear_model import OLS, WLS
from statsmodels.tools.tools import add_constant
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro, jarque_bera
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.gofplots import qqplot
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.filters.hp_filter import hpfilter
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import r2_score
import statsmodels.api as sm


class CointegrationTester:
    def __init__(self, df, variables):
        self.df = df
        self.variables = variables

    def run_engle_granger(self):
        results = []
        for var1, var2 in combinations(self.variables, 2):
            s1 = self.df[var1].dropna()
            s2 = self.df[var2].dropna()
            common_idx = s1.index.intersection(s2.index)
            stat, pval, _ = coint(s1.loc[common_idx], s2.loc[common_idx])
            results.append({
                "Série 1": var1,
                "Série 2": var2,
                "Stat cointégration": stat,
                "p-value": pval,
                "Cointégré": pval < 0.05
            })
        return pd.DataFrame(results).sort_values("p-value")

    def run_johansen(self, k_ar_diff=1, det_order=0, group_size=3):
        results = []
        for group in combinations(self.variables, group_size):
            data = self.df[list(group)].dropna()
            if len(data) < 30:
                continue
            joh = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)
            trace_stats = joh.lr1
            cvt_95 = joh.cvt[:, 1]
            r = sum(trace_stats > cvt_95)
            results.append({
                "Variables": group,
                "Nombre relations cointégrées (5%)": r
            })
        return pd.DataFrame(results)

    def compute_combination_and_test_adf(self, selected_vars, beta_index=0):
        data = self.df[selected_vars].dropna()
        joh = coint_johansen(data, det_order=0, k_ar_diff=1)
        beta = joh.evec[:, beta_index]
        self.df["z_coint"] = data.values @ beta
        pval = adfuller(self.df["z_coint"].dropna())[1]
        return beta, pval

class WLSModelEvaluator:
    def __init__(self, df, features, targets):
        self.df = df.copy()
        self.features = features
        self.targets = targets
        self.models = {}
        self.residuals = {}

    def fit_models(self):
        for target in self.targets:
            df_clean = self.df.dropna(subset=self.features + [target])
            X = add_constant(df_clean[self.features])
            y = df_clean[target]
            model = sm.WLS(y, X).fit()
            self.models[target] = model
            self.residuals[target] = y - model.predict(X)

    def display_results(self):
        for target, model in self.models.items():
            print(f"\n--- Résultats pour {target} ---")
            print(model.summary())

    def plot_diagnostics(self):
        for target, model in self.models.items():
            df_clean = self.df.dropna(subset=self.features + [target])
            X = add_constant(df_clean[self.features])
            y = df_clean[target]
            y_pred = model.predict(X)
            residuals = y - y_pred

            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f"Diagnostics - {target}")

            axs[0, 0].scatter(y_pred, residuals, alpha=0.6)
            axs[0, 0].axhline(0, color='red', linestyle='--')
            axs[0, 0].set_title("Résidus vs Prédictions")

            sns.histplot(residuals, kde=True, ax=axs[0, 1], bins=20)
            axs[0, 1].set_title("Distribution des résidus")

            plot_acf(residuals, ax=axs[1, 0], lags=20)
            axs[1, 0].set_title("Autocorrélation des résidus")

            qqplot(residuals, line='s', ax=axs[1, 1])
            axs[1, 1].set_title("QQ-plot des résidus")

            plt.tight_layout()
            plt.show()

            print("Durbin-Watson:", durbin_watson(residuals))
            print("Breusch-Pagan p-val:", het_breuschpagan(residuals, X)[1])
            print("Shapiro p-val:", shapiro(residuals)[1])
            print("Jarque-Bera p-val:", jarque_bera(residuals)[1])


def test_pls_on_segments(df, macro_vars, segments):
    print("=== PLS Regression Test ===")
    for i in range(1, 6):
        pls = PLSRegression(n_components=2)
        X = df[macro_vars].dropna()
        y = df[f'Indicateur_moyen_Brut_{i}'].loc[X.index]
        pls.fit(X, y)
        df['PLS1'] = pls.transform(X)[:, 0]
        df['PLS2'] = pls.transform(X)[:, 1]
        y_pred = pls.predict(X)
        r2 = r2_score(y, y_pred)
        print(f"Segment {i} - R²: {r2:.4f}")


def compare_models_on_segment(df, macro_vars, target_col):
    X = df[macro_vars].dropna()
    y = df[target_col].loc[X.index]
    results = []

    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X, y)
    y_pred = rf.predict(X)
    res = y - y_pred
    results.append({
        "Modèle": "Random Forest",
        "R²": round(r2_score(y, y_pred), 4),
        "Shapiro p-val": round(shapiro(res).pvalue, 4),
        "Shapiro": "Normaux" if shapiro(res).pvalue > 0.05 else "Non normaux",
        "Jarque-Bera p-val": round(jarque_bera(res).pvalue, 4),
        "Jarque-Bera": "Normaux" if jarque_bera(res).pvalue > 0.05 else "Non normaux"
    })

    huber = HuberRegressor()
    huber.fit(X, y)
    y_pred = huber.predict(X)
    res = y - y_pred
    results.append({
        "Modèle": "Huber Regressor",
        "R²": round(r2_score(y, y_pred), 4),
        "Shapiro p-val": round(shapiro(res).pvalue, 4),
        "Shapiro": "Normaux" if shapiro(res).pvalue > 0.05 else "Non normaux",
        "Jarque-Bera p-val": round(jarque_bera(res).pvalue, 4),
        "Jarque-Bera": "Normaux" if jarque_bera(res).pvalue > 0.05 else "Non normaux"
    })

    mod = sm.QuantReg(y, sm.add_constant(X)).fit(q=0.5)
    y_pred = mod.predict(sm.add_constant(X))
    res = y - y_pred
    results.append({
        "Modèle": "Quantile Regression (q=0.5)",
        "R²": round(r2_score(y, y_pred), 4),
        "Shapiro p-val": round(shapiro(res).pvalue, 4),
        "Shapiro": "Normaux" if shapiro(res).pvalue > 0.05 else "Non normaux",
        "Jarque-Bera p-val": round(jarque_bera(res).pvalue, 4),
        "Jarque-Bera": "Normaux" if jarque_bera(res).pvalue > 0.05 else "Non normaux"
    })

    return pd.DataFrame(results)


def run_segment_models(df, macro_vars, segments):
    results_model = []
    results_modelW = []

    for seg in segments:
        X = df[macro_vars].copy()
        X = sm.add_constant(X)
        try:
            y = df[seg + '_diff']
        except:
            y = df[seg]

        segment_num = seg.split('_')[-1]
        weights = df.get(f'PourcNoteCohorte5_{segment_num}', pd.Series([1] * len(df), index=df.index))

        best_r2 = -np.inf
        best_model = None
        best_k = None

        best_r2w = -np.inf
        best_modelW = None
        best_kw = None

        for k in range(2, len(macro_vars) + 1):
            try:
                corrs = X.drop(columns='const').apply(lambda col: np.corrcoef(col, y)[0, 1])
                X_top_cols = corrs.abs().sort_values(ascending=False).head(k).index.tolist()
                X_top_corr = sm.add_constant(X[X_top_cols], has_constant='add')

                model = sm.OLS(y, X_top_corr).fit()
                modelW = sm.WLS(y, X_top_corr, weights=weights).fit()

                if model.rsquared > best_r2:
                    best_r2 = model.rsquared
                    best_model = model
                    best_k = k

                if modelW.rsquared > best_r2w:
                    best_r2w = modelW.rsquared
                    best_modelW = modelW
                    best_kw = k

            except Exception as e:
                print(f"Erreur sur {seg}, k={k} : {e}")
                continue

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

    return pd.DataFrame(results_model), pd.DataFrame(results_modelW)
