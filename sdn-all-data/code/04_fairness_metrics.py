#!/usr/bin/env python3
# Step 4 - fairness audit of Random Forest predictions (Dataset A vs Dataset B).
# Trains Random Forest on an 80/20 split and computes SPD, DI, EOD (one-vs-rest per controller, worst case over classes)
# and Individual Fairness (k-NN, k=5) with topology as the sensitive attribute.
# Writes a JSON result file and a text report.
import csv, os, sys, json, logging, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
import traceback

warnings.filterwarnings('ignore')

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_A = os.path.join(DATA_DIR, 'dataset_A_ahp.csv')
DATASET_A_CLEAN = os.path.join(DATA_DIR, 'dataset_clean_after_cl.csv')
DATASET_B = os.path.join(DATA_DIR, 'dataset_B_ground_truth.csv')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

FAIRNESS_CONFIG = {
    'privileged_groups': ['linear', 'star', 'tree'],
    'disadvantaged_groups': ['mesh', 'ring'],
    'sensitive_attr': 'topology_type'
}

FEATURE_COLS = ['throughput_mbps', 'latency_ms', 'packet_loss_percent',
                'jitter_ms', 'cpu_usage_percent', 'memory_usage_mb',
                'flow_setup_time_ms', 'nodes']


def load_dataset(filepath):
    if not os.path.exists(filepath):
        logger.error(f"Dataset not found: {filepath}")
        return None
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} samples from {os.path.basename(filepath)}")
    return df


def calculate_individual_fairness(y_pred, X_scaled, k=5):
    n = len(y_pred)
    nn = NearestNeighbors(n_neighbors=min(k+1, n), algorithm='ball_tree')
    nn.fit(X_scaled)
    _, indices = nn.kneighbors(X_scaled)

    weighted_mismatch = 0.0
    total_weight = 0.0

    for i in range(n):
        for j_idx in range(1, min(k+1, n)):
            j = indices[i, j_idx]
            w_ij = 1.0
            weighted_mismatch += abs(int(y_pred[i]) - int(y_pred[j])) * w_ij
            total_weight += w_ij

    if total_weight == 0:
        return 1.0, {'n_pairs': 0}

    if_score = 1.0 - (weighted_mismatch / total_weight)
    return float(if_score), {'n_pairs': int(total_weight)}

def calculate_fairness_metrics(df_test, sensitive_col, y_true, y_pred,
                                class_names, X_test_scaled, y_proba,
                                di_epsilon=0):
    privileged_mask = df_test[sensitive_col].isin(
        FAIRNESS_CONFIG['privileged_groups']).values
    disadvantaged_mask = df_test[sensitive_col].isin(
        FAIRNESS_CONFIG['disadvantaged_groups']).values

    n_priv = int(privileged_mask.sum())
    n_dis = int(disadvantaged_mask.sum())

    per_class = {}
    for class_idx, class_name in enumerate(class_names):
        y_true_fav = (y_true == class_idx).astype(int)
        y_pred_fav = (y_pred == class_idx).astype(int)

        p_fav_priv = y_pred_fav[privileged_mask].mean() if n_priv > 0 else 0
        p_fav_dis = y_pred_fav[disadvantaged_mask].mean() if n_dis > 0 else 0
        spd = abs(p_fav_dis - p_fav_priv)

        di = (p_fav_dis + di_epsilon) / (p_fav_priv + di_epsilon)

        priv_true_pos = privileged_mask & (y_true_fav == 1)
        dis_true_pos = disadvantaged_mask & (y_true_fav == 1)
        tpr_priv = y_pred_fav[priv_true_pos].mean() if priv_true_pos.sum() > 0 else 0
        tpr_dis = y_pred_fav[dis_true_pos].mean() if dis_true_pos.sum() > 0 else 0
        eod = abs(tpr_dis - tpr_priv)

        per_class[class_name] = {
            'p_favorable_privileged': round(float(p_fav_priv), 4),
            'p_favorable_disadvantaged': round(float(p_fav_dis), 4),
            'spd': round(float(spd), 4),
            'di': round(float(di), 4),
            'eod': round(float(eod), 4),
            'tpr_privileged': round(float(tpr_priv), 4),
            'tpr_disadvantaged': round(float(tpr_dis), 4),
            'spd_fair': bool(spd < 0.1),
            'di_fair': bool(0.8 <= di <= 1.2),
            'eod_fair': bool(eod < 0.1),
        }

    spd_by_class = {c: m['spd'] for c, m in per_class.items()}
    eod_by_class = {c: m['eod'] for c, m in per_class.items()}
    di_by_class = {c: m['di'] for c, m in per_class.items()}

    spd_worst_class = max(spd_by_class, key=spd_by_class.get)
    eod_worst_class = max(eod_by_class, key=eod_by_class.get)
    spd_agg = spd_by_class[spd_worst_class]
    eod_agg = eod_by_class[eod_worst_class]
    spd_mean = round(float(np.mean(list(spd_by_class.values()))), 4)
    eod_mean = round(float(np.mean(list(eod_by_class.values()))), 4)

    di_worst_class = max(di_by_class, key=lambda c: abs(di_by_class[c] - 1))
    di_agg = di_by_class[di_worst_class]

    individual_fairness, if_details = calculate_individual_fairness(
        y_pred, X_test_scaled, k=5
    )

    spd_fair = bool(spd_agg < 0.1)
    di_fair = bool(0.8 <= di_agg <= 1.2)
    eod_fair = bool(eod_agg < 0.1)
    individual_fair = bool(individual_fairness > 0.9)

    fair_count = sum([spd_fair, di_fair, eod_fair, individual_fair])

    primary_fair = bool(eod_fair and individual_fair)
    primary_fair_count = sum([eod_fair, individual_fair])

    metrics = {
        'n_privileged': n_priv,
        'n_disadvantaged': n_dis,
        'per_class': per_class,
        'spd': round(float(spd_agg), 4),
        'spd_worst_class': spd_worst_class,
        'spd_mean_across_classes': spd_mean,
        'di': round(float(di_agg), 4),
        'di_worst_class': di_worst_class,
        'di_epsilon_used': di_epsilon,
        'eod': round(float(eod_agg), 4),
        'eod_worst_class': eod_worst_class,
        'eod_mean_across_classes': eod_mean,
        'individual_fairness': round(float(individual_fairness), 4),
        'if_n_pairs': if_details.get('n_pairs'),
        'if_L_used': if_details.get('L_used'),
        'if_empirical_L_max_ratio': if_details.get('empirical_L_max_ratio'),
        'if_violation_rate': if_details.get('violation_rate'),
        'spd_fair': spd_fair,
        'di_fair': di_fair,
        'eod_fair': eod_fair,
        'individual_fair': individual_fair,
        'fair_count': fair_count,
        'primary_fair': primary_fair,
        'primary_fair_count': primary_fair_count,
        'primary_criteria_note': (
            "Kriteria utama = EOD + IF (bukan SPD/DI). SPD/DI dilaporkan "
            "sebagai pelengkap deskriptif: nilai tinggi berpotensi "
            "mencerminkan pola topologi->kontroler yang legitimate "
            "(lihat label_distribution_percent), bukan murni bias sistem. "
            "Perlu dijustifikasi/didiskusikan eksplisit di Bab 3/4."
        ),
    }

    logger.info(f"  [SPD] worst-case={spd_agg:.4f} (kelas: {spd_worst_class}) | "
                f"mean={spd_mean:.4f} | Fair: {spd_fair} [pelengkap]")
    logger.info(f"  [DI]  worst-case={di_agg:.4f} (eps={di_epsilon}) "
                f"(kelas: {di_worst_class}) | Fair: {di_fair} [pelengkap]")
    logger.info(f"  [EOD] worst-case={eod_agg:.4f} (kelas: {eod_worst_class}) | "
                f"mean={eod_mean:.4f} | Fair: {eod_fair} [KRITERIA UTAMA]")
    logger.info(f"  [IF]  {individual_fairness:.4f} | Fair: {individual_fair} [KRITERIA UTAMA] "
                f"(n_pairs={if_details.get('n_pairs')})")
    for c, m in per_class.items():
        logger.info(f"    - {c}: SPD={m['spd']:.4f} DI={m['di']:.4f} EOD={m['eod']:.4f}")
    logger.info(f"  Fairness metrics achieved (4 metrik/Tabel 3.6): {fair_count}/4")
    logger.info(f"  Primary fair (EOD+IF): {primary_fair_count}/2 -> {primary_fair}")

    return metrics


def evaluate_model(df, dataset_name):
    if 'topology_type' not in df.columns:
        logger.error("topology_type column missing")
        return None
    if 'best_controller_label' not in df.columns:
        logger.error("best_controller_label column missing")
        return None

    df_features = df[FEATURE_COLS].fillna(0)
    scaler = StandardScaler()
    X = scaler.fit_transform(df_features)

    le = LabelEncoder()
    y = le.fit_transform(df['best_controller_label'])

    indices = np.arange(len(df))

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, test_size=0.20, random_state=42
    )

    df_test = df.iloc[idx_test].reset_index(drop=True)

    results = {
        'dataset': dataset_name,
        'n_samples': len(df),
        'n_features': len(FEATURE_COLS),
        'n_train': len(X_train),
        'n_test': len(X_test),
    }

    rf_model = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1
    )
    try:
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        y_proba_rf = rf_model.predict_proba(X_test)

        results['rf_accuracy'] = round(accuracy_score(y_test, y_pred_rf), 4)
        results['rf_precision'] = round(precision_score(y_test, y_pred_rf, average='weighted', zero_division=0), 4)
        results['rf_recall'] = round(recall_score(y_test, y_pred_rf, average='weighted', zero_division=0), 4)
        results['rf_f1'] = round(f1_score(y_test, y_pred_rf, average='weighted', zero_division=0), 4)

        logger.info(f"{dataset_name} - RF Accuracy:  {results['rf_accuracy']:.4f}")
        logger.info(f"{dataset_name} - RF Precision: {results['rf_precision']:.4f}")
        logger.info(f"{dataset_name} - RF Recall:    {results['rf_recall']:.4f}")
        logger.info(f"{dataset_name} - RF F1-Score:  {results['rf_f1']:.4f}")

        logger.info(f"\n--- Fairness Metrics ({dataset_name}, berbasis prediksi RF, one-vs-rest) ---")
        fairness = calculate_fairness_metrics(
            df_test, FAIRNESS_CONFIG['sensitive_attr'],
            y_test, y_pred_rf, le.classes_, X_test, y_proba_rf
        )
        results.update(fairness)

    except Exception as e:
        logger.error(f"RF evaluation error: {e}")
        results['error'] = str(e)

    dt_model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    try:
        dt_model.fit(X_train, y_train)
        y_pred_dt = dt_model.predict(X_test)

        results['dt_accuracy'] = round(accuracy_score(y_test, y_pred_dt), 4)
        results['dt_precision'] = round(precision_score(y_test, y_pred_dt, average='weighted', zero_division=0), 4)
        results['dt_recall'] = round(recall_score(y_test, y_pred_dt, average='weighted', zero_division=0), 4)
        results['dt_f1'] = round(f1_score(y_test, y_pred_dt, average='weighted', zero_division=0), 4)

        logger.info(f"{dataset_name} - DT Accuracy (baseline pembanding): {results['dt_accuracy']:.4f}")

    except Exception as e:
        logger.error(f"DT evaluation error: {e}")
        results['dt_error'] = str(e)

    return results


def compare_datasets():
    logger.info("=" * 60)
    logger.info("DATASET EVALUATION: Comparing Dataset A vs Dataset B")
    logger.info("=" * 60)

    df_a = load_dataset(DATASET_A)
    df_b = load_dataset(DATASET_B)

    if df_a is None:
        logger.error("Dataset A not found.")
        return

    eval_results = []

    logger.info("\n--- Dataset A (AHP Labels) ---")
    results_a = evaluate_model(df_a, "Dataset A")
    if results_a:
        eval_results.append(results_a)

    if df_b is not None:
        logger.info("\n--- Dataset B (Ground Truth: CL + iFlipper) ---")
        results_b = evaluate_model(df_b, "Dataset B")
        if results_b:
            eval_results.append(results_b)

        if results_a and results_b and 'spd' in results_a and 'spd' in results_b:
            logger.info("\n--- Fairness Improvement (RF predictions) ---")
            improvement_spd = abs(results_a['spd']) - abs(results_b['spd'])
            improvement_eod = abs(results_a['eod']) - abs(results_b['eod'])
            logger.info(f"SPD |A| - |B|: {improvement_spd:+.4f} [pelengkap]")
            logger.info(f"EOD |A| - |B|: {improvement_eod:+.4f} [KRITERIA UTAMA]")
            logger.info(f"Fair metrics achieved (4 metrik/Tabel 3.6): {results_a['fair_count']}/4 -> {results_b['fair_count']}/4")
            logger.info(f"Primary fair (EOD+IF): {results_a['primary_fair_count']}/2 -> {results_b['primary_fair_count']}/2")
    else:
        logger.warning("Dataset B not found - skipping comparison")

    results_file = os.path.join(OUTPUT_DIR, '04_fairness_results.json')
    with open(results_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")

    return eval_results


def generate_summary_report():
    results_file = os.path.join(OUTPUT_DIR, '04_fairness_results.json')
    if not os.path.exists(results_file):
        logger.error("No evaluation results found")
        return

    with open(results_file, 'r') as f:
        results = json.load(f)

    report = []
    report.append("=" * 60)
    report.append("THESIS EVALUATION SUMMARY REPORT")
    report.append("=" * 60)

    for res in results:
        report.append(f"\n--- {res.get('dataset', 'Unknown')} ---")
        report.append(f"  Samples: {res.get('n_samples', 'N/A')} (train={res.get('n_train','N/A')}, test={res.get('n_test','N/A')})")
        report.append(f"  Features: {res.get('n_features', 'N/A')}")

        if 'rf_accuracy' in res:
            report.append(f"  [Random Forest] Accuracy: {res['rf_accuracy']:.4f} | "
                           f"Precision: {res['rf_precision']:.4f} | "
                           f"Recall: {res['rf_recall']:.4f} | "
                           f"F1: {res['rf_f1']:.4f}")
        if 'dt_accuracy' in res:
            report.append(f"  [Decision Tree - baseline tambahan] Accuracy: {res['dt_accuracy']:.4f} | "
                           f"Precision: {res['dt_precision']:.4f} | "
                           f"Recall: {res['dt_recall']:.4f} | "
                           f"F1: {res['dt_f1']:.4f}")

        if 'spd' in res:
            di_str = f"{res['di']:.4f}" if np.isfinite(res['di']) else str(res['di'])
            report.append(f"  [Pelengkap] SPD: {res['spd']:.4f} worst-class={res.get('spd_worst_class')} (fair={res['spd_fair']}) | "
                           f"DI: {di_str} worst-class={res.get('di_worst_class')} (fair={res['di_fair']}, eps={res.get('di_epsilon_used')})")
            report.append(f"  [Kriteria utama] EOD: {res['eod']:.4f} worst-class={res.get('eod_worst_class')} (fair={res['eod_fair']}) | "
                           f"IF: {res['individual_fairness']:.4f} (fair={res['individual_fair']})")
            report.append(f"  Fairness metrics achieved (4 metrik/Tabel 3.6): {res.get('fair_count','N/A')}/4")
            report.append(f"  Primary fair (EOD+IF): {res.get('primary_fair_count','N/A')}/2 -> {res.get('primary_fair')}")
            if 'per_class' in res:
                report.append("  Rincian per kelas (one-vs-rest):")
                for cls_name, cls_metrics in res['per_class'].items():
                    di_cls = cls_metrics['di']
                    di_cls_str = f"{di_cls:.4f}" if isinstance(di_cls, (int, float)) and np.isfinite(di_cls) else str(di_cls)
                    report.append(
                        f"    - {cls_name}: SPD={cls_metrics['spd']:.4f} "
                        f"DI={di_cls_str} EOD={cls_metrics['eod']:.4f}"
                    )

    report_text = "\n".join(report)
    print(report_text)

    report_file = os.path.join(OUTPUT_DIR, '04_fairness_report.txt')
    with open(report_file, 'w') as f:
        f.write(report_text)
    logger.info(f"Report saved to {report_file}")


def main():
    logger.info("Starting evaluation...")
    compare_datasets()
    generate_summary_report()
    logger.info("Evaluation complete")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
