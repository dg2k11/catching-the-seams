"""Evaluation metrics: binary Macro-F1, Slice H false-positive rate, and AI-share MAE."""


def compute_f1_for_class(y_true, y_pred, target):
    if not y_true:
        return 0.0
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == target and yp == target)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != target and yp == target)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == target and yp != target)
    denom = 2 * tp + fp + fn
    if denom == 0:
        # Absent class with zero errors scores 1.0 so Slice H pure-human episodes get full terminal bonus.
        return 1.0
    return (2.0 * tp) / denom


def compute_macro_f1(y_true, y_pred):
    return 0.5 * (compute_f1_for_class(y_true, y_pred, 0) + compute_f1_for_class(y_true, y_pred, 1))


def compute_slice_h_fpr(y_pred_slice_h):
    if not y_pred_slice_h:
        return 0.0
    return sum(1 for yp in y_pred_slice_h if yp == 1) / len(y_pred_slice_h)


def compute_passage_share_mae(true_shares, pred_shares):
    if not true_shares:
        return 0.0
    return sum(abs(a - b) for a, b in zip(true_shares, pred_shares)) / len(true_shares)
