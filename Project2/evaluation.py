"""Evaluation Metrics

Author: Kristina Striegnitz and Claudia Porto

I affirm that I have carried out my academic endeavors with full
academic honesty. Claudia Porto

Complete this file for part 1 of the project.
"""

def get_accuracy(y_pred, y_true):
    """Calculate the accuracy of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    try:
        tp = true_pos_count(y_pred, y_true)
        tn = true_neg_count(y_pred, y_true)
        num_correct = tp + tn
        return num_correct / len(y_true)
    except:
        return None


def get_precision(y_pred, y_true):
    """Calculate the precision of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    try:
        tp = true_pos_count(y_pred, y_true)
        fp = false_pos_count(y_pred, y_true)
        return tp / (tp + fp)
    except:
        return None


def get_recall(y_pred, y_true):
    """Calculate the recall of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    try:
        tp = true_pos_count(y_pred, y_true)
        fn = false_neg_count(y_pred, y_true)
        return tp / (tp + fn)
    except:
        return None



def get_fscore(y_pred, y_true):
    """Calculate the f-score of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    try:
        precision = get_precision(y_pred, y_true)
        recall = get_recall(y_pred, y_true)
        return (2 * precision * recall) / (precision + recall)
    except:
        return None


def evaluate(y_pred, y_true):
    """Calculate precision, recall, and f-score of the predicted labels
    and print out the results.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    accuracy = get_accuracy(y_pred, y_true)
    precision = get_precision(y_pred, y_true)
    recall = get_recall(y_pred, y_true)
    fscore = get_fscore(y_pred, y_true)

    print('Accuracy: {:.0%}'.format(accuracy))
    print('Precision: {:.0%}'.format(precision))
    print('Recall: {:.0%}'.format(recall))
    print('F-score: {:.0%}'.format(fscore))


##### PRIVATE HELPER METHODS ######

def true_pos_count(y_pred, y_true):
    all_pred_true = zip(y_pred, y_true)
    tp_count = 0
    for pred, true in all_pred_true:
        if pred == 1 and true == 1:
            tp_count += 1
    return tp_count

def true_neg_count(y_pred, y_true):
    all_pred_true = zip(y_pred, y_true)
    tn_count = 0
    for pred, true in all_pred_true:
        if pred == 0 and true == 0:
            tn_count += 1
    return tn_count

def false_pos_count(y_pred, y_true):
    all_pred_true = zip(y_pred, y_true)
    fp_count = 0
    for pred, true in all_pred_true:
        if pred > true:
            fp_count += 1
    return fp_count

def false_neg_count(y_pred, y_true):
    all_pred_true = zip(y_pred, y_true)
    fn_count = 0
    for pred, true in all_pred_true:
        if pred < true:
            fn_count += 1
    return fn_count
