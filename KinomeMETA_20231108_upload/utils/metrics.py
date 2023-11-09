import math
from typing import List, Callable, Union

from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score, mean_absolute_error, r2_score, \
    precision_recall_curve, auc, recall_score, precision_score, confusion_matrix, matthews_corrcoef, \
    balanced_accuracy_score, f1_score
from scipy.stats import pearsonr, spearmanr

optimization_direction = {
    'roc_auc': 1,
    'prc_auc': 1,
    'rmse': -1,
    'mae': -1,
    'r2': 1,
    "pearson": 1,
    "spearman": 1,
    'accuracy': 1,
    'recall': 1,
    'sensitivity': 1,
    'specificity': 1,
    'matthews_corrcoef': 1,
    'f1': 1
}


def accuracy(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:
    """
    Computes the accuracy of a binary prediction task using a given threshold for generating hard predictions.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0
    :return: The computed accuracy.
    """
    hard_preds = [1 if p > threshold else 0 for p in preds]
    return accuracy_score(targets, hard_preds)


def balanced_accuracy(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:
    hard_preds = [1 if p > threshold else 0 for p in preds]
    return balanced_accuracy_score(targets, hard_preds)


def recall(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:
    """
    Computes the recall of a binary prediction task using a given threshold for generating hard predictions.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0
    :return: The computed recall.
    """
    hard_preds = [1 if p > threshold else 0 for p in preds]
    return recall_score(targets, hard_preds)


def precision(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:
    """
    Computes the recall of a binary prediction task using a given threshold for generating hard predictions.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0
    :return: The computed recall.
    """
    hard_preds = [1 if p > threshold else 0 for p in preds]
    return precision_score(targets, hard_preds)


def sensitivity(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:
    """
    Computes the sensitivity of a binary prediction task using a given threshold for generating hard predictions.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0
    :return: The computed sensitivity.
    """
    return recall(targets, preds, threshold)


def specificity(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:
    """
    Computes the specificity of a binary prediction task using a given threshold for generating hard predictions.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0
    :return: The computed specificity.
    """
    hard_preds = [1 if p > threshold else 0 for p in preds]
    tn, fp, fn, tp = confusion_matrix(targets, hard_preds).ravel()
    return tn / float(tn + fp)


def mcc(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:
    """
    Computes the Matthews correlation coefficient of a binary prediction task using a given threshold for generating hard predictions.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0
    :return: The computed Matthews correlation coefficient.
    """
    hard_preds = [1 if p > threshold else 0 for p in preds]
    return matthews_corrcoef(targets, hard_preds)

def f1(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:
    hard_preds = [1 if p > threshold else 0 for p in preds]
    return f1_score(targets, hard_preds)


def rmse(targets: List[float], preds: List[float]) -> float:
    """
    Computes the root mean squared error.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed rmse.
    """
    try:
        return math.sqrt(mean_squared_error(targets, preds))
    except ValueError:
        return float('nan')


def mae(targets: List[float], preds: List[float]) -> float:
    try:
        return mean_absolute_error(targets, preds)
    except ValueError:
        return float('nan')

def r2(targets: List[float], preds: List[float]) -> float:
    try:
        return r2_score(targets, preds)
    except ValueError:
        return float('nan')


def roc(targets: List[float], preds: List[float]) -> float:
    try:
        return roc_auc_score(targets, preds)
    except ValueError:
        return float('nan')


def pearson(targets: List[float], preds: List[float]) -> float:
    """
    Computes the root mean squared error.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed rmse.
    """
    try:
        return pearsonr(targets, preds)[0]
    except ValueError:
        return float('nan')


def spearman(targets: List[float], preds: List[float]) -> float:
    """
    Computes the root mean squared error.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed rmse.
    """
    try:
        return spearmanr(targets, preds)[0]
    except ValueError:
        return float('nan')

def bacc(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:
    """
    Computes the Matthews correlation coefficient of a binary prediction task using a given threshold for generating hard predictions.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0
    :return: The computed Matthews correlation coefficient.
    """
    try:
        hard_preds = [1 if p > threshold else 0 for p in preds]
        return balanced_accuracy_score(targets, hard_preds)
    except ValueError:
        return float('nan')


def get_metric_func(metric: str) -> Callable[[Union[List[int], List[float]], List[float]], float]:
    """
    Gets the metric function corresponding to a given metric name.

    :param metric: Metric name.
    :return: A metric function which takes as arguments a list of targets and a list of predictions and returns.
    """
    # Note: If you want to add a new metric, please also update the parser argument --metric in parsing.py.
    if metric == 'roc_auc':
        return roc

    if metric == 'prc_auc':
        return prc_auc

    if metric == 'f1-score':
        return f1

    if metric == 'rmse':
        return rmse

    if metric == 'mae':
        return mae

    if metric == 'r2':
        return r2

    if metric == 'pearson':
        return pearson

    if metric == 'spearman':
        return spearman

    if metric == 'accuracy':
        return accuracy

    if metric == 'balanced_accuracy':
        return balanced_accuracy

    if metric == 'recall':
        return recall

    if metric == 'precision':
        return precision

    if metric == 'sensitivity':
        return sensitivity

    if metric == 'specificity':
        return specificity

    if metric == 'matthews_corrcoef':
        return mcc

    if metric == 'bacc':
        return bacc

    raise ValueError(f'Metric "{metric}" not supported.')


def prc_auc(targets: List[int], preds: List[float]) -> float:
    """
    Computes the area under the precision-recall curve.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :return: The computed prc-auc.
    """
    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)
