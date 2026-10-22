#!/usr/bin/env python3
# Step 3 - accuracy comparison and significance tests (Dataset A vs Dataset B).
# Runs 5-fold StratifiedKFold (no shuffling) with Random Forest, Decision Tree and SVM, reports weighted F1 per fold, then a
# two-sided paired t-test, 95% CI, Cohen's d and a Wilcoxon signed-rank test on the per-fold differences.
# Writes a JSON result file and a per-fold CSV.

import os, sys, json, logging, warnings, traceback
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR)
DATA_DIR   = os.path.join(ROOT_DIR, 'data')
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


def preprocess(df, le):
    X = df[FEATURE_COLS].fillna(0).values
    X = StandardScaler().fit_transform(X)
    y = le.transform(df[LABEL_COL])
    return X, y


def cv_f1(model, X, y):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=False)
    return cross_val_score(clone(model), X, y, cv=cv,
                           scoring='f1_weighted', n_jobs=N_JOBS)


def run_paired_test_single(model, model_name, X_a, y_a, X_b, y_b):
    scores_a = cv_f1(model, X_a, y_a)
    scores_b = cv_f1(model, X_b, y_b)

    d     = scores_b - scores_a
    n     = len(d)
    df    = n - 1
    d_bar = float(d.mean())
    s_d   = float(d.std(ddof=1))

    t_stat, p_value = stats.ttest_rel(scores_b, scores_a, alternative='two-sided')

    t_crit = stats.t.ppf(1 - (1 - CONF_LEVEL) / 2, df)
    se     = s_d / np.sqrt(n)
    ci_low, ci_high = d_bar - t_crit * se, d_bar + t_crit * se

    cohens_d = d_bar / s_d if s_d > 0 else float('nan')

    try:
        w_stat, w_p = stats.wilcoxon(scores_b, scores_a, alternative='two-sided')
        w_stat, w_p = float(w_stat), float(w_p)
    except ValueError:
        w_stat, w_p = float('nan'), float('nan')

    n_positive = int((d > 0).sum())
    improvement = d_bar > 0
    p_below_alpha = bool(p_value < ALPHA)

    logger.info(f"\n  [{model_name}]")
    logger.info(f"  F1 per fold A     : {np.round(scores_a, 4)}")
    logger.info(f"  F1 per fold B     : {np.round(scores_b, 4)}")
    logger.info(f"  Selisih per fold  : {np.round(d, 4)}  ({n_positive}/{n} fold positif)")
    logger.info(f"  D_bar={d_bar:.4f}, s_D={s_d:.4f}, t({df})={t_stat:.4f}, p={p_value:.4f}")
    logger.info(f"  95% CI D_bar      : [{ci_low:.4f}, {ci_high:.4f}]")
    logger.info(f"  Cohen's d (d_z)   : {cohens_d:.4f}")
    logger.info(f"  Wilcoxon          : W={w_stat}, p={w_p:.4f}")

    return {
        'model'        : model_name,
        'n_pairs'      : n,
        'df'           : df,
        'scores_a'     : [round(float(s), 4) for s in scores_a],
        'scores_b'     : [round(float(s), 4) for s in scores_b],
        'diff_per_fold': [round(float(x), 4) for x in d],
        'mean_a'       : round(float(scores_a.mean()), 4),
        'std_a'        : round(float(scores_a.std(ddof=STD_DDOF)), 4),
        'mean_b'       : round(float(scores_b.mean()), 4),
        'std_b'        : round(float(scores_b.std(ddof=STD_DDOF)), 4),
        'd_bar'        : round(d_bar, 4),
        's_d'          : round(s_d, 4),
        't_stat'       : round(float(t_stat), 4),
        'p_value'      : round(float(p_value), 4),
        'ci95_low'     : round(float(ci_low), 4),
        'ci95_high'    : round(float(ci_high), 4),
        'cohens_d'     : round(float(cohens_d), 4),
        'wilcoxon_stat': w_stat,
        'wilcoxon_p'   : round(w_p, 4),
        'n_positive_folds': n_positive,
        'p_below_alpha': p_below_alpha,
        'improvement'  : bool(improvement),
        'criterion_met': bool(p_below_alpha and improvement),
    }


def run_all_models(df_a, df_b, le):
    logger.info(f"\n=== UJI SIGNIFIKANSI ({N_SPLITS}-Fold CV, paired t-test dua sisi) ===")
    X_a, y_a = preprocess(df_a, le)
    X_b, y_b = preprocess(df_b, le)
    return {name: run_paired_test_single(m, name, X_a, y_a, X_b, y_b)
            for name, m in MODELS.items()}


def evaluate_descriptive_all(df, dataset_name, le):
    X, y = preprocess(df, le)
    results = {'dataset': dataset_name, 'n_samples': len(df)}
    for model_name, model in MODELS.items():
        scores = cv_f1(model, X, y)
        results[model_name] = {
            'f1_mean': round(float(scores.mean()), 4),
            'f1_std' : round(float(scores.std(ddof=STD_DDOF)), 4),
        }
        logger.info(f"  {dataset_name} | {model_name}: "
                    f"{scores.mean():.4f} ± {scores.std(ddof=STD_DDOF):.4f}")
    return results


def main():
    logger.info("=" * 70)
    logger.info("EVALUASI — RF, DT, SVM: paired t-test, 95% CI, Cohen's d, Wilcoxon")
    logger.info("=" * 70)

    df_a = pd.read_csv(DATASET_A)
    df_b = pd.read_csv(DATASET_B)
    logger.info(f"Dataset A (AHP) : {len(df_a)} sampel")
    logger.info(f"Dataset B (GT)  : {len(df_b)} sampel")
    logger.info("Catatan: pemasangan skor hanya berdasarkan indeks fold "
                "(data uji kedua fold tidak identik).")

    le = LabelEncoder().fit(df_a[LABEL_COL])
    logger.info(f"Kelas label     : {list(le.classes_)}")

    logger.info("\n=== EVALUASI DESKRIPTIF ===")
    desc_a = evaluate_descriptive_all(df_a, "Dataset A (AHP)", le)
    desc_b = evaluate_descriptive_all(df_b, "Dataset B (GT)", le)

    results = run_all_models(df_a, df_b, le)

    out_json = os.path.join(OUTPUT_DIR, '03_statistical_validation_results.json')
    with open(out_json, 'w') as f:
        json.dump({'descriptive_a': desc_a, 'descriptive_b': desc_b,
                   'significance_tests': results}, f, indent=2)

    rows = []
    for name, r in results.items():
        for i in range(r['n_pairs']):
            rows.append({'model': name, 'fold': i + 1,
                         'f1_dataset_a': r['scores_a'][i],
                         'f1_dataset_b': r['scores_b'][i],
                         'diff_b_minus_a': r['diff_per_fold'][i]})
    out_csv = os.path.join(OUTPUT_DIR, '03_per_fold_differences.csv')
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    logger.info("\n" + "=" * 70)
    logger.info("RINGKASAN UNTUK TABEL 4.10")
    logger.info("=" * 70)
    header = (f"{'Model':<14}{'F1 A':<16}{'F1 B':<16}{'D_bar':>8}{'t(4)':>9}"
              f"{'p':>8}{'95% CI D_bar':>22}{'d':>8}{'p Wilcoxon':>12}{'Kriteria':>10}")
    logger.info(header)
    logger.info("-" * len(header))
    for name, r in results.items():
        ci = f"[{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]"
        logger.info(
            f"{name:<14}"
            f"{r['mean_a']:.4f}±{r['std_a']:.4f}  "
            f"{r['mean_b']:.4f}±{r['std_b']:.4f}  "
            f"{r['d_bar']:>8.4f}{r['t_stat']:>9.4f}{r['p_value']:>8.4f}"
            f"{ci:>22}{r['cohens_d']:>8.4f}{r['wilcoxon_p']:>12.4f}"
            f"{'Terpenuhi' if r['criterion_met'] else 'Tidak':>10}"
        )
    logger.info(f"\nHasil lengkap       : {out_json}")
    logger.info(f"Selisih per fold    : {out_csv}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
