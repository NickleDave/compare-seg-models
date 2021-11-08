import numpy as np


def levenshtein(source, target):
    """levenshtein distance
    returns number of deletions, insertions, or substitutions
    required to convert source string into target string.

    Parameters
    ----------
    source : str
        in this context, predicted labels for songbird syllables
    target : str
        in this context, ground truth labels for songbird syllables

    Returns
    -------
    distance : int
        number of deletions, insertions, or substitutions
        required to convert source into target.

    from https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
    """

    if len(source) < len(target):
        return levenshtein(target, source)

    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = np.array(tuple(source))
    target = np.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
                current_row[1:],
                np.add(previous_row[:-1], target != s))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
                current_row[1:],
                current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1]


def syllable_error_rate(true, pred):
    """syllable error rate: word error rate, but with songbird syllables
    Levenshtein/edit distance normalized by length of true sequence

    Parameters
    ----------
    true : str
        ground truth labels for a series of songbird syllables
    pred : str
        predicted labels for a series of songbird syllables

    Returns
    -------
    Levenshtein distance / len(true)
    """

    if type(true) != str or type(pred) != str:
        raise TypeError('Both `true` and `pred` must be of type `str')

    return levenshtein(pred, true) / len(true)


# metric from
# [1]René, Colin Lea Michael D. Flynn, and Vidal Austin Reiter Gregory D. Hager.
# "Temporal convolutional networks for action segmentation and detection." (2017).
# https://github.com/colincsl/TemporalConvolutionalNetworks
def overlap_f1(P, Y, n_classes=0, bg_class=None, overlap=.1, **kwargs):
    def overlap_(p, y, n_classes, bg_class, overlap):

        true_intervals = np.array(utils.segment_intervals(y))
        true_labels = utils.segment_labels(y)
        pred_intervals = np.array(utils.segment_intervals(p))
        pred_labels = utils.segment_labels(p)

        # Remove background labels
        if bg_class is not None:
            true_intervals = true_intervals[true_labels != bg_class]
            true_labels = true_labels[true_labels != bg_class]
            pred_intervals = pred_intervals[pred_labels != bg_class]
            pred_labels = pred_labels[pred_labels != bg_class]

        n_true = true_labels.shape[0]
        n_pred = pred_labels.shape[0]

        # We keep track of the per-class TPs, and FPs.
        # In the end we just sum over them though.
        TP = np.zeros(n_classes, np.float)
        FP = np.zeros(n_classes, np.float)
        true_used = np.zeros(n_true, np.float)

        for j in range(n_pred):
            # Compute IoU against all others
            intersection = np.minimum(pred_intervals[j, 1],
                                      true_intervals[:, 1]) - np.maximum(
                pred_intervals[j, 0], true_intervals[:, 0])
            union = np.maximum(pred_intervals[j, 1],
                               true_intervals[:, 1]) - np.minimum(
                pred_intervals[j, 0], true_intervals[:, 0])
            IoU = (intersection / union) * (pred_labels[j] == true_labels)

            # Get the best scoring segment
            idx = IoU.argmax()

            # If the IoU is high enough and the true segment isn't already used
            # Then it is a true positive. Otherwise is it a false positive.
            if IoU[idx] >= overlap and not true_used[idx]:
                TP[pred_labels[j]] += 1
                true_used[idx] = 1
            else:
                FP[pred_labels[j]] += 1

        TP = TP.sum()
        FP = FP.sum()
        # False negatives are any unused true segment (i.e. "miss")
        FN = n_true - true_used.sum()

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * (precision * recall) / (precision + recall)

        # If the prec+recall=0, it is a NaN. Set these to 0.
        F1 = np.nan_to_num(F1)

        return F1 * 100

    if type(P) == list:
        return np.mean(
            [overlap_(P[i], Y[i], n_classes, bg_class, overlap) for i in
             range(len(P))])
    else:
        return overlap_(P, Y, n_classes, bg_class, overlap)