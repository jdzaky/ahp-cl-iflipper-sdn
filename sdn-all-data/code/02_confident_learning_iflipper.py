#!/usr/bin/env python3
# Step 2 - Confident Learning + iFlipper pipeline (Dataset A -> Dataset B).
# Removes noisy AHP labels with group-wise Confident Learning (privileged vs disadvantaged topologies, iterative),
# then flips labels with a greedy multi-class iFlipper on a k-NN similarity graph (k=5, error budget 10%).
# Writes the cleaned dataset, the ground truth (Dataset B, with an iflipper_applied flag) and a JSON summary.
import csv, os, sys, json, logging, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linprog
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


def load_dataset(filepath):
    if not os.path.exists(filepath):
        logger.error(f"Dataset not found: {filepath}")
        return None
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} samples from {os.path.basename(filepath)}")
    return df


def psi(x):
    return np.log(1 + x + (x ** 2) / 2)


def compute_threshold(probs, labels, n_classes, Ns, nu):
    thresholds = {}
    for j in range(n_classes):
        idx = np.where(labels == j)[0]
        if len(idx) == 0:
            thresholds[j] = 0.0
            continue

        p_j = probs[idx, j]
        t_bar = np.mean(psi(p_j))

        Q = nu * (Ns + (nu * np.log(2 * Ns)) / (Ns ** 2))
        mu_j = t_bar - Q / (Ns - nu)

        thresholds[j] = max(mu_j, 0.0)
    return thresholds


def assign_true_labels(probs_A, probs_B, labels, thresh_A, thresh_B, n_classes, s):
    N = len(labels)
    z_hat = np.full(N, -1, dtype=int)
    source = np.full(N, '', dtype=object)

    for n in range(N):
        y_n = labels[n]

        if s[n] == 1:
            p = probs_B[n, y_n]
            if p >= thresh_B.get(y_n, 0):
                z_hat[n] = int(np.argmax(probs_B[n]))
                source[n] = 'B'
            else:
                source[n] = 'none'
        else:
            p = probs_A[n, y_n]
            if p >= thresh_A.get(y_n, 0):
                z_hat[n] = int(np.argmax(probs_A[n]))
                source[n] = 'A'
            else:
                source[n] = 'none'

    return z_hat, source


def fair_confident_learning(df, epochs=50, Ns_fraction=0.6, nu=1e-2):
    logger.info("=" * 60)
    logger.info("FAIR CONFIDENT LEARNING (REVISED - Iterative Progressive)")
    logger.info("=" * 60)

    feature_cols = ['throughput_mbps', 'latency_ms', 'packet_loss_percent',
                  'jitter_ms', 'cpu_usage_percent', 'memory_usage_mb',
                  'flow_setup_time_ms', 'nodes']

    df_copy = df.copy()
    df_features = df_copy[feature_cols].fillna(0)

    scaler = StandardScaler()
    X = scaler.fit_transform(df_features)

    le = LabelEncoder()
    y = le.fit_transform(df_copy['best_controller_label'])
    n_classes = len(le.classes_)

    s = df_copy[FAIRNESS_CONFIG['sensitive_attr']].isin(
        FAIRNESS_CONFIG['privileged_groups']
    ).astype(int).values

    N = len(y)
    Ns = max(int(Ns_fraction * N), 2)

    idx_A = np.where(s == 0)[0]
    idx_B = np.where(s == 1)[0]

    logger.info(f"Total samples: {N}")
    logger.info(f"Privileged (B): {len(idx_B)} | Disadvantaged (A): {len(idx_A)}")
    logger.info(f"Estimated clean: {Ns}")

    clean_mask = np.ones(N, dtype=bool)
    all_biased = set()

    for epoch in range(epochs):
        current_clean = np.where(clean_mask)[0]

        clean_A = np.intersect1d(current_clean, idx_A)
        clean_B = np.intersect1d(current_clean, idx_B)

        clf_A = LogisticRegression(max_iter=1000, random_state=42)
        clf_B = LogisticRegression(max_iter=1000, random_state=42)

        if len(clean_A) >= n_classes and len(np.unique(y[clean_A])) >= n_classes:
            clf_A.fit(X[clean_A], y[clean_A])
        else:
            clf_A.fit(X[idx_A], y[idx_A])

        if len(clean_B) >= n_classes and len(np.unique(y[clean_B])) >= n_classes:
            clf_B.fit(X[clean_B], y[clean_B])
        else:
            clf_B.fit(X[idx_B], y[idx_B])

        all_classes = np.arange(n_classes)

        def predict_proba_full(clf, X_input, all_classes):
            raw = clf.predict_proba(X_input)
            full = np.zeros((len(X_input), len(all_classes)))
            for i, c in enumerate(clf.classes_):
                full[:, c] = raw[:, i]
            return full

        probs_A = predict_proba_full(clf_A, X, all_classes)
        probs_B = predict_proba_full(clf_B, X, all_classes)

        thresh_A = compute_threshold(probs_A, y, n_classes, Ns, nu)
        thresh_B = compute_threshold(probs_B, y, n_classes, Ns, nu)

        z_hat, source = assign_true_labels(probs_A, probs_B, y, thresh_A, thresh_B, n_classes, s)

        off_diag_noisy = set()
        for n in range(N):
            if z_hat[n] != -1 and y[n] != z_hat[n]:
                off_diag_noisy.add(n)

        epoch_noisy = off_diag_noisy

        new_noisy = epoch_noisy - all_biased
        all_biased.update(epoch_noisy)

        clean_mask = np.ones(N, dtype=bool)
        clean_mask[list(all_biased)] = False

        if (epoch + 1) % 10 == 0 or epoch < 5:
            logger.info(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"New noisy: {len(new_noisy):4d} | "
                  f"Total noisy: {len(all_biased):4d} | "
                  f"Clean remaining: {np.sum(clean_mask):4d}")

        if len(new_noisy) == 0:
            logger.info(f"Converged at epoch {epoch+1}")
            break

    df_copy['is_noisy_cl'] = False
    df_copy.loc[list(all_biased), 'is_noisy_cl'] = True

    noisy_count = len(all_biased)
    logger.info(f"\nConfident Learning detected {noisy_count} noisy samples ({noisy_count/N*100:.1f}%)")
    logger.info(f"Clean samples remaining: {N - noisy_count}")

    return df_copy


def generate_similarity_matrix(X, method='knn', k=5):
    n = len(X)

    if method == 'knn':
        nn = NearestNeighbors(n_neighbors=min(k+1, n), algorithm='ball_tree')
        nn.fit(X)
        _, indices = nn.kneighbors(X)

        W = np.zeros((n, n))
        for i in range(n):
            for j_idx in range(1, min(k+1, n)):
                j = indices[i, j_idx]
                W[i, j] = 1.0

        W = np.maximum(W, W.T)
        np.fill_diagonal(W, 0)

    return W


def compute_weighted_mismatches(y, W):
    n = len(y)
    total_error = 0.0
    edge_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] > 0:
                edge_count += 1
                if y[i] != y[j]:
                    total_error += W[i, j]

    return total_error, edge_count


def compute_flip_impact(y, W, idx, n_classes):
    n = len(y)
    current_label = y[idx]
    impacts = {}

    for new_label in range(n_classes):
        if new_label == current_label:
            continue

        new_error = 0.0
        for j in range(n):
            if W[idx, j] > 0:
                if new_label != y[j]:
                    new_error += W[idx, j]

        old_error = 0.0
        for j in range(n):
            if W[idx, j] > 0:
                if current_label != y[j]:
                    old_error += W[idx, j]

        reduction = old_error - new_error
        impacts[new_label] = reduction

    return impacts


def iflipper_multiclass(y, W, error_budget, n_classes):
    n = len(y)
    y_flipped = y.copy()

    initial_error, edge_count = compute_weighted_mismatches(y_flipped, W)
    error_budget_val = initial_error * error_budget

    logger.info(f"iFlipper Multi-Class: Initial error={initial_error:.2f}, "
                f"edges={edge_count}, budget={error_budget_val:.2f} ({error_budget*100:.0f}%)")

    if initial_error <= error_budget_val:
        logger.info("Error already within budget, no flips needed.")
        return y_flipped, 0

    flips_done = 0
    max_iterations = n * n_classes

    for iteration in range(max_iterations):
        current_error, _ = compute_weighted_mismatches(y_flipped, W)

        if current_error <= error_budget_val:
            break

        best_reduction = 0
        best_idx = -1
        best_new_label = -1

        for idx in range(n):
            impacts = compute_flip_impact(y_flipped, W, idx, n_classes)

            for new_label, reduction in impacts.items():
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_idx = idx
                    best_new_label = new_label

        if best_reduction <= 0 or best_idx == -1:
            logger.info(f"No positive reduction found at iteration {iteration+1}")
            break

        old_label = y_flipped[best_idx]
        y_flipped[best_idx] = best_new_label
        flips_done += 1

        if (flips_done % 50 == 0) or (iteration < 10):
            new_error, _ = compute_weighted_mismatches(y_flipped, W)
            logger.info(f"  Flip {flips_done}: idx={best_idx} {old_label}->{best_new_label} "
                      f"(reduction={best_reduction:.3f}), error={new_error:.2f}")

    final_error, _ = compute_weighted_mismatches(y_flipped, W)
    logger.info(f"iFlipper: {flips_done} labels flipped, error {initial_error:.2f}->{final_error:.2f}")

    return y_flipped, flips_done


def apply_iflipper_fairness(df, error_fraction=0.1):
    logger.info("=" * 60)
    logger.info("IFLIPPER: Individual Fairness (REVISED - Multi-Class Greedy)")
    logger.info("=" * 60)

    feature_cols = ['throughput_mbps', 'latency_ms', 'packet_loss_percent',
                   'jitter_ms', 'cpu_usage_percent', 'memory_usage_mb',
                   'flow_setup_time_ms', 'nodes']

    df_fair = df.copy()
    df_features = df_fair[feature_cols].fillna(0)

    scaler = StandardScaler()
    X = scaler.fit_transform(df_features)

    le = LabelEncoder()
    le.fit(df_fair['best_controller_label'].unique())
    y_original = le.transform(df_fair['best_controller_label'])
    n_classes = len(le.classes_)

    logger.info(f"Classes: {le.classes_} (n={n_classes})")
    logger.info(f"Generating similarity matrix for {len(X)} samples...")
    W = generate_similarity_matrix(X, method='knn', k=5)

    y_flipped, total_flips = iflipper_multiclass(y_original, W, error_fraction, n_classes)

    df_fair['best_controller_label'] = le.inverse_transform(y_flipped)

    df_fair['iflipper_applied'] = (y_original != y_flipped)

    logger.info(f"iFlipper: {total_flips} labels modified for fairness")

    return df_fair


def apply_confident_learning_filter(df):
    df_cl = fair_confident_learning(df, epochs=50, Ns_fraction=0.6, nu=1e-2)
    return df_cl


def apply_iflipper_correction(df):
    return apply_iflipper_fairness(df, error_fraction=0.1)


def main():
    logger.info("=" * 60)
    logger.info("PIPELINE: CONFIDENT LEARNING + IFLIPPER (REVISED)")
    logger.info("=" * 60)

    df = load_dataset(DATASET_A)
    if df is None:
        logger.error("Please ensure dataset exists at: " + DATASET_A)
        return

    logger.info(f"Original dataset: {len(df)} samples")

    logger.info("\n" + "=" * 60)
    logger.info("Stage 1: CONFIDENT LEARNING (REVISED)")
    logger.info("=" * 60)
    df_cl = apply_confident_learning_filter(df)

    df_clean = df_cl[~df_cl.get('is_noisy_cl', pd.Series([False]*len(df_cl)))].copy()
    if 'is_noisy_cl' in df_clean.columns:
        df_clean = df_clean.drop(columns=['is_noisy_cl'])

    noisy_removed = len(df) - len(df_clean)
    logger.info(f"Removed {noisy_removed} noisy samples, kept {len(df_clean)}")

    df_clean.to_csv(DATASET_A_CLEAN, index=False)
    logger.info(f"Clean dataset saved: {DATASET_A_CLEAN}")

    logger.info("\n" + "=" * 60)
    logger.info("Stage 2: IFLIPPER (REVISED Multi-Class)")
    logger.info("=" * 60)
    df_fair = apply_iflipper_correction(df_clean)

    df_fair.to_csv(DATASET_B, index=False)
    logger.info(f"Fair dataset saved: {DATASET_B}")

    pipeline_results = {
        'original_samples': len(df),
        'clean_samples': len(df_clean),
        'fair_samples': len(df_fair),
        'noise_removed': noisy_removed,
    }

    results_file = os.path.join(OUTPUT_DIR, '02_pipeline_summary.json')
    with open(results_file, 'w') as f:
        json.dump(pipeline_results, f, indent=2)
    logger.info(f"\nResults saved: {results_file}")

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Original (AHP): {pipeline_results['original_samples']}")
    logger.info(f"After CL:       {pipeline_results['clean_samples']}")
    logger.info(f"After iFlipper: {pipeline_results['fair_samples']}")

    return pipeline_results


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
