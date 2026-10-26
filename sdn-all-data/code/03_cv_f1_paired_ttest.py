#!/usr/bin/env python3
# Step 3 - classification performance and paired t-tests (Dataset A vs Dataset B).
# Runs 5-fold StratifiedKFold (no shuffling) with Random Forest, Decision Tree and SVM and reports, per fold:
# accuracy, macro-precision, macro-recall and weighted F1. For every model and metric it then runs a two-sided
# paired t-test on the per-fold differences (Dataset B minus Dataset A) with a 95% confidence interval.
# The 12 p-values (3 models x 4 metrics) form one family and are adjusted with the Holm step-down method.
# Writes a JSON result file, a per-fold CSV, a summary CSV and LaTeX table rows.

import os, sys, json, logging, warnings, traceback
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (make_scorer, accuracy_score, precision_score,
                             recall_score, f1_score)
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR)
DATA_DIR   = os.path.join(ROOT_DIR, 'code/data')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_A = os.path.join(DATA_DIR, 'dataset_A_ahp.csv')
DATASET_B = os.path.join(DATA_DIR, 'dataset_B_ground_truth.csv')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
N_ESTIMATORS = 100
N_SPLITS     = 5
N_JOBS       = -1
ALPHA        = 0.05
CONF_LEVEL   = 0.95
# Standard deviation of the fold scores: 1 = sample SD (n-1), 0 = population SD (n).
# The earlier version of this script used 1; the paper table must use the same convention.
STD_DDOF     = 1

FEATURE_COLS = [
    'throughput_mbps', 'latency_ms', 'packet_loss_percent',
    'jitter_ms', 'cpu_usage_percent', 'memory_usage_mb',
    'flow_setup_time_ms', 'nodes'
]
LABEL_COL = 'best_controller_label'

MODELS = {
    'Random Forest': RandomForestClassifier(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=N_JOBS
    ),
    'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
    'SVM': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=RANDOM_STATE),
}

# key -> (display name, scorer). zero_division=0 makes a class that is never predicted count as 0.
METRICS = {
    'accuracy'       : ('Accuracy',
                        make_scorer(accuracy_score)),
    'precision_macro': ('Macro-precision',
                        make_scorer(precision_score, average='macro', zero_division=0)),
    'recall_macro'   : ('Macro-recall',
                        make_scorer(recall_score, average='macro', zero_division=0)),
    'f1_weighted'    : ('Weighted F1',
                        make_scorer(f1_score, average='weighted', zero_division=0)),
}
SCORERS = {key: spec[1] for key, spec in METRICS.items()}

def holm_adjust(p_values):
    """Holm step-down adjusted p-values (same order as the input).
    Sorted ascending p_(1) <= ... <= p_(m): adj_(i) = max_{j<=i} min(1, (m - j + 1) * p_(j))."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):                 # rank = 0 for the smallest p-value
        candidate = min(1.0, (m - rank) * p_values[idx])
        running_max = max(running_max, candidate)
        adjusted[idx] = running_max
    return adjusted

def preprocess(df, le):
    X = df[FEATURE_COLS].fillna(0).values
    X = StandardScaler().fit_transform(X)
    y = le.transform(df[LABEL_COL])
    return X, y

def cv_all_metrics(model, X, y):
    """Return {metric_key: array of per-fold scores} from one 5-fold run."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=False)
    res = cross_validate(model, X, y, cv=cv, scoring=SCORERS, n_jobs=N_JOBS)
    return {key: np.asarray(res['test_' + key], dtype=float) for key in METRICS}

def paired_test(scores_a, scores_b):
    d     = scores_b - scores_a
    n     = len(d)
    df    = n - 1
    d_bar = float(d.mean())
    s_d   = float(d.std(ddof=1))            # sample SD of the differences, as in the t statistic

    t_stat, p_value = stats.ttest_rel(scores_b, scores_a, alternative='two-sided')

    t_crit = stats.t.ppf(1 - (1 - CONF_LEVEL) / 2, df)
    se     = s_d / np.sqrt(n)
    ci_low, ci_high = d_bar - t_crit * se, d_bar + t_crit * se

    return {
        'n_pairs'         : n,
        'df'              : df,
        'scores_a'        : [round(float(s), 4) for s in scores_a],
        'scores_b'        : [round(float(s), 4) for s in scores_b],
        'diff_per_fold'   : [round(float(x), 4) for x in d],
        'mean_a'          : round(float(scores_a.mean()), 4),
        'std_a'           : round(float(scores_a.std(ddof=STD_DDOF)), 4),
        'mean_b'          : round(float(scores_b.mean()), 4),
        'std_b'           : round(float(scores_b.std(ddof=STD_DDOF)), 4),
        'd_bar'           : round(d_bar, 4),
        's_d'             : round(s_d, 4),
        't_stat'          : round(float(t_stat), 4),
        'p_value'         : round(float(p_value), 4),
        'p_raw'           : float(p_value),     # unrounded, used only for the Holm adjustment
        'ci95_low'        : round(float(ci_low), 4),
        'ci95_high'       : round(float(ci_high), 4),
        'n_positive_folds': int((d > 0).sum()),
        'improvement'     : bool(d_bar > 0),
    }

def run_all(df_a, df_b, le):
    X_a, y_a = preprocess(df_a, le)
    X_b, y_b = preprocess(df_b, le)
    results = {}
    for model_name, model in MODELS.items():
        cv_a = cv_all_metrics(model, X_a, y_a)
        cv_b = cv_all_metrics(model, X_b, y_b)
        results[model_name] = {}
        for key, (label, _) in METRICS.items():
            r = paired_test(cv_a[key], cv_b[key])
            r['metric'] = label
            results[model_name][key] = r

    # ---- Holm step-down over the whole family of 3 models x 4 metrics = 12 tests
    index = [(m, k) for m in results for k in results[m]]
    adjusted = holm_adjust([results[m][k]['p_raw'] for m, k in index])
    for (m, k), p_adj in zip(index, adjusted):
        r = results[m][k]
        r['p_holm']           = round(float(p_adj), 4)
        r['significant_raw']  = bool(r['p_raw'] < ALPHA)
        r['significant_holm'] = bool(p_adj < ALPHA)
        del r['p_raw']

    for model_name, per_metric in results.items():
        logger.info(f"\n  [{model_name}]")
        for r in per_metric.values():
            logger.info(f"  {r['metric']:<16} A={r['mean_a']:.4f}±{r['std_a']:.4f}  "
                        f"B={r['mean_b']:.4f}±{r['std_b']:.4f}  "
                        f"D_bar={r['d_bar']:.4f}  t({r['df']})={r['t_stat']:.4f}  "
                        f"p={r['p_value']:.4f}  p_Holm={r['p_holm']:.4f}  "
                        f"95% CI=[{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]  "
                        f"({r['n_positive_folds']}/{r['n_pairs']} folds up)")
    return results

def main():
    logger.info("=" * 70)
    logger.info("EVALUASI - RF, DT, SVM: accuracy, macro-precision, macro-recall, weighted F1")
    logger.info("Paired t-test (two-sided) + 95% CI, koreksi Holm atas 12 uji")
    logger.info("=" * 70)

    df_a = pd.read_csv(DATASET_A)
    df_b = pd.read_csv(DATASET_B)
    logger.info(f"Dataset A (AHP) : {len(df_a)} sampel")
    logger.info(f"Dataset B (GT)  : {len(df_b)} sampel")
    logger.info("Catatan: pemasangan skor hanya berdasarkan indeks fold "
                "(data uji kedua fold tidak identik).")
    logger.info(f"SD skor fold    : ddof={STD_DDOF}")

    le = LabelEncoder().fit(df_a[LABEL_COL])
    logger.info(f"Kelas label     : {list(le.classes_)}")

    logger.info(f"\n=== {N_SPLITS}-FOLD CV, 4 METRIK x 3 MODEL ===")
    results = run_all(df_a, df_b, le)

    # ---- JSON
    out_json = os.path.join(OUTPUT_DIR, '03_metrics_paired_ttest_results.json')
    with open(out_json, 'w') as f:
        json.dump({'std_ddof': STD_DDOF, 'alpha': ALPHA, 'conf_level': CONF_LEVEL,
                   'multiplicity_correction': 'Holm step-down, family = 12 tests '
                                              '(3 models x 4 metrics); CI is not adjusted',
                   'results': results}, f, indent=2)

    # ---- per-fold CSV (long format) and summary rows
    fold_rows, summary_rows = [], []
    for model_name, per_metric in results.items():
        for key, r in per_metric.items():
            for i in range(r['n_pairs']):
                fold_rows.append({'model': model_name, 'metric': r['metric'], 'fold': i + 1,
                                  'dataset_a': r['scores_a'][i],
                                  'dataset_b': r['scores_b'][i],
                                  'diff_b_minus_a': r['diff_per_fold'][i]})
            summary_rows.append({'model': model_name, 'metric': r['metric'],
                                 'mean_a': r['mean_a'], 'std_a': r['std_a'],
                                 'mean_b': r['mean_b'], 'std_b': r['std_b'],
                                 'd_bar': r['d_bar'], 't_stat': r['t_stat'],
                                 'df': r['df'], 'p_value': r['p_value'],
                                 'p_holm': r['p_holm'],
                                 'ci95_low': r['ci95_low'], 'ci95_high': r['ci95_high']})
    out_folds   = os.path.join(OUTPUT_DIR, '03_per_fold_scores.csv')
    out_summary = os.path.join(OUTPUT_DIR, '03_metrics_summary.csv')
    pd.DataFrame(fold_rows).to_csv(out_folds, index=False)
    pd.DataFrame(summary_rows).to_csv(out_summary, index=False)

    # ---- LaTeX rows for the paper (Model & Metric & A & B & d_bar & t(4) & p & p_Holm & 95% CI)
    out_tex = os.path.join(OUTPUT_DIR, '03_latex_rows.tex')
    with open(out_tex, 'w') as f:
        for s in summary_rows:
            f.write(f"{s['model']} & {s['metric']} & "
                    f"{s['mean_a']:.4f} $\\pm$ {s['std_a']:.4f} & "
                    f"{s['mean_b']:.4f} $\\pm$ {s['std_b']:.4f} & "
                    f"{s['d_bar']:.4f} & {s['t_stat']:.4f} & {s['p_value']:.4f} & "
                    f"{s['p_holm']:.4f} & "
                    f"[{s['ci95_low']:.4f}; {s['ci95_high']:.4f}] \\\\\n\\hline\n")

    # ---- console summary
    logger.info("\n" + "=" * 70)
    logger.info("RINGKASAN (D_bar = Dataset B - Dataset A)")
    logger.info("=" * 70)
    header = (f"{'Model':<14}{'Metrik':<17}{'A (mean±SD)':<17}{'B (mean±SD)':<17}"
              f"{'D_bar':>8}{'t(4)':>9}{'p':>8}{'p_Holm':>9}{'95% CI':>22}")
    logger.info(header)
    logger.info("-" * len(header))
    for s in summary_rows:
        ci = f"[{s['ci95_low']:.4f}, {s['ci95_high']:.4f}]"
        logger.info(f"{s['model']:<14}{s['metric']:<17}"
                    f"{s['mean_a']:.4f}±{s['std_a']:.4f}   "
                    f"{s['mean_b']:.4f}±{s['std_b']:.4f}   "
                    f"{s['d_bar']:>8.4f}{s['t_stat']:>9.4f}{s['p_value']:>8.4f}"
                    f"{s['p_holm']:>9.4f}{ci:>22}")
    n_raw  = sum(r['significant_raw']  for pm in results.values() for r in pm.values())
    n_holm = sum(r['significant_holm'] for pm in results.values() for r in pm.values())
    logger.info(f"\nUji dengan p < {ALPHA}: {n_raw}/12 tanpa koreksi, {n_holm}/12 setelah koreksi Holm.")
    logger.info("Sanity check: weighted F1 harus sama dengan hasil lama "
                "(RF 0.4061 -> 0.8224, DT 0.4252 -> 0.7464, SVM 0.4033 -> 0.8196).")
    logger.info(f"\nHasil lengkap : {out_json}")
    logger.info(f"Skor per fold : {out_folds}")
    logger.info(f"Ringkasan     : {out_summary}")
    logger.info(f"Baris LaTeX   : {out_tex}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
