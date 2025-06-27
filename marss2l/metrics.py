
from sklearn.metrics import accuracy_score, balanced_accuracy_score, \
    precision_score, recall_score, average_precision_score, log_loss, \
    confusion_matrix
from numpy.typing import NDArray
from typing import Dict
import numpy as np


def get_scenelevel_metrics(scene_pred_cont:NDArray, target:NDArray, threshold:float=0.5,
                           as_percentage:bool=False) -> Dict[str, float]:
    preds_discrete = (scene_pred_cont > threshold).astype(int)

    precision = precision_score(target, preds_discrete)
    recall = recall_score(target, preds_discrete)
    accuracy = accuracy_score(target, preds_discrete)
    balanced_accuracy = balanced_accuracy_score(target, preds_discrete)
    avr_precision = average_precision_score(target, scene_pred_cont)

    bce = log_loss(target, scene_pred_cont, labels=[0, 1])
    fpr_value = fpr(target, preds_discrete)

    if as_percentage:
        precision *= 100
        recall *= 100
        accuracy *= 100
        balanced_accuracy *= 100
        avr_precision *= 100
        # bce *= 100
        fpr_value *= 100

    return {"average_precision": float(avr_precision),
            "precision": float(precision),
            "recall": float(recall),
            "accuracy": float(accuracy),
            "binary_cross_entropy": float(bce),
            "fpr": float(fpr_value),
            "balanced_accuracy": float(balanced_accuracy)}

def fpr(y_true:NDArray, y_pred:NDArray) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return float(cm[0, 1] / cm[0].sum())

def get_pixellevel_metrics(TP:NDArray, TN:NDArray, 
                           FP:NDArray, FN:NDArray,
                           mode:str="micro",
                           as_percentage:bool=False) -> Dict[str, float]:
    """
    Compute pixel level metrics from confusion matrix of samples.

    Args:
        TP (NDArray): Array of True Positives (of size number of samples)
        TN (NDArray): Array of True Negatives (of size number of samples)
        FP (NDArray): Array of False Positives (of size number of samples)
        FN (NDArray): Array of False Negatives (of size number of samples)
        mode (str, optional): Mode of the metrics. Defaults to "macro".
            If "macro", the metrics are averaged across samples.
            If "micro", the totals are added and the metrics are computed.
        as_percentage (bool, optional): If True, the metrics are returned as percentages. 
            Defaults to False. If true, the metrics are multiplied by 100.

    Returns:
        Dict[str, float]: Dictionary of metrics: "precision", "recall", "accuracy", "fpr", "f1", "intersection_over_union"
    """
    assert mode in ["macro", "micro"], "Mode must be either 'macro' or 'micro'"

    if mode == "macro":
        precision = np.mean(TP / (TP + FP))
        recall = np.mean(TP / (TP + FN))
        accuracy = np.mean((TP + TN) / (TP + TN + FP + FN))
        fpr = np.mean(FP / (FP + TN))
        f1 = np.mean(2 * precision * recall / (precision + recall))
        intersection_over_union = np.mean(TP / (TP + FP + FN))
    else:
        precision = np.sum(TP) / np.sum(TP + FP)
        recall = np.sum(TP) / np.sum(TP + FN)
        fpr = np.sum(FP) / np.sum(FP + TN)
        accuracy = np.sum(TP + TN) / np.sum(TP + TN + FP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        intersection_over_union = np.sum(TP) / np.sum(TP + FP + FN)
    
    if as_percentage:
        precision *= 100
        recall *= 100
        accuracy *= 100
        fpr *= 100
        f1 *= 100
        intersection_over_union *= 100

    return {"segmentation_precision": float(precision),
            "segmentation_recall": float(recall),
            "segmentation_accuracy": float(accuracy),
            "segmentation_f1": float(f1),
            "segmentation_fpr": float(fpr),
            "iou": float(intersection_over_union)}
