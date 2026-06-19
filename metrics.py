import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    confusion_matrix,
)


def build_siamese_pairs(features: np.ndarray, labels: np.ndarray, num_pairs: int = 2000, seed: int = 42):
    """Genera coppie bilanciate di esempi simili e diversi per la valutazione."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    x1 = np.zeros((num_pairs, features.shape[1]), dtype=features.dtype)
    x2 = np.zeros_like(x1)
    similarity_labels = np.zeros(num_pairs, dtype=np.int64)

    for i in range(num_pairs):
        idx = int(rng.integers(n))
        if rng.random() < 0.5:
            same_indices = np.where(labels == labels[idx])[0]
            partner = int(rng.choice(same_indices))
            similarity_labels[i] = 1
        else:
            different_indices = np.where(labels != labels[idx])[0]
            partner = int(rng.choice(different_indices))
            similarity_labels[i] = 0

        x1[i] = features[idx]
        x2[i] = features[partner]

    return x1, x2, similarity_labels


def compute_pairwise_distances(model, x1: np.ndarray, x2: np.ndarray, device: str = "cpu", batch_size: int = 256):
    """Calcola le distanze euclidee tra le coppie nello spazio latente del modello."""
    model.eval()
    distances = []

    with torch.no_grad():
        for start in range(0, len(x1), batch_size):
            end = start + batch_size
            batch_x1 = torch.FloatTensor(x1[start:end]).to(device)
            batch_x2 = torch.FloatTensor(x2[start:end]).to(device)
            emb1, emb2 = model(batch_x1, batch_x2)
            batch_dist = torch.nn.functional.pairwise_distance(emb1, emb2, eps=1e-6)
            distances.append(batch_dist.cpu().numpy())

    return np.concatenate(distances, axis=0)


def select_best_distance_threshold(distances: np.ndarray, labels: np.ndarray):
    """Scorre la curva ROC per trovare la soglia che massimizza il trade-off TPR-FPR."""
    if len(np.unique(labels)) < 2:
        return float(np.median(distances))

    y_scores = -distances
    fpr, tpr, thresholds = roc_curve(labels, y_scores)
    best_index = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_index]
    return float(-best_threshold)


def compute_pairwise_metrics(distances: np.ndarray, labels: np.ndarray, threshold: float = 1.0):
    """Restituisce metriche classiche per la classificazione di coppie simili/diverse."""
    predictions = (distances < threshold).astype(int)
    y_scores = -distances

    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, y_scores) if len(np.unique(labels)) > 1 else np.nan,
        "average_precision": average_precision_score(labels, y_scores) if len(np.unique(labels)) > 1 else np.nan,
        "mean_distance_similar": float(np.mean(distances[labels == 1])) if np.any(labels == 1) else np.nan,
        "mean_distance_different": float(np.mean(distances[labels == 0])) if np.any(labels == 0) else np.nan,
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }

    return metrics


def compute_policy_safety_metrics(true_labels: np.ndarray, exclusion_predictions: np.ndarray):
    """Calcola le metriche di sicurezza della policy basate su esclusione / non esclusione."""
    true_labels = np.asarray(true_labels).astype(int)
    exclusion_predictions = np.asarray(exclusion_predictions).astype(bool)

    tp = np.logical_and(true_labels == 1, exclusion_predictions).sum()
    tn = np.logical_and(true_labels == 0, ~exclusion_predictions).sum()
    fp = np.logical_and(true_labels == 0, exclusion_predictions).sum()
    fn = np.logical_and(true_labels == 1, ~exclusion_predictions).sum()

    metrics = {
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        "exclusion_rate": float(exclusion_predictions.mean()),
    }

    return metrics
