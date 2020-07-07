# Utilities for the Data Preparation
# This function standardize the data from (-128 to 127) to (-1 to 1) to help Network training. Try without this also

import numpy as np
from keras import backend as K

# set parameters
sample_size = 800000

# min max values of the signal
max_num = 127
min_num = -128
# ------------------------------------------------------------------------------------
def min_max_scaling(ts, min_data, max_data, range_needed=(-1, 1)):
    if min_data < 0:
        ts_std = (ts + abs(min_data)) / (max_data + abs(min_data))
    else:
        ts_std = (ts - min_data) / (max_data - min_data)
    if range_needed[0] < 0:
        return ts_std * (range_needed[1] + abs(range_needed[0])) + range_needed[0]
    else:
        return ts_std * (range_needed[1] - range_needed[0]) + range_needed[0]


def transform_ts(ts, n_dim=160, min_max=(-1, 1)):

    # convert data into -1 to 1

    ts_std = min_max_scaling(ts, min_data=min_num, max_data=max_num)
    bucket_size = int(sample_size / n_dim)

    new_ts = []
    for i in range(0, sample_size, bucket_size):

        ts_range = ts_std[i:i + bucket_size]

        # calculate each feature
        mean = ts_range.mean()
        std = ts_range.std()  # standard deviation
        std_top = mean + std  # I have to test it more, but is is like a band
        std_bot = mean - std

        # percentile feats
        percentil_calc = np.percentile(ts_range, [0, 1, 25, 50, 75, 99, 100])
        max_range = percentil_calc[-1] - percentil_calc[0]  # this is the amplitude of the chunk
        relative_percentile = percentil_calc - mean  # maybe it could heap to understand the asymmetry

        # collate all features
        new_ts.append(np.concatenate([np.asarray([mean, std, std_top, std_bot, max_range]), percentil_calc, relative_percentile]))

    return np.asarray(new_ts)

# evaluation metric
# It is the official metric used in this competition
# below is the declaration of a function used inside the keras model, calculation with K (keras backend / thensorflow)
def matthews_correlation(y_true, y_pred):

    """
    Calculates the Matthews correlation coefficient measure for quality
    of binary classification problems.
    """

    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


# The output of this kernel must be binary (0 or 1), but the output of the NN Model is float (0 to 1).
# So, find the best threshold to convert float to binary is crucial to the result
# this piece of code is a function that evaluates all the possible thresholds from 0 to 1 by 0.01
def threshold_search(y_true, y_prob):
    best_threshold = 0
    best_score = 0
    for threshold in [i * 0.01 for i in range(100)]:
        score = K.eval(matthews_correlation(y_true.astype(np.float64), (y_prob > threshold).astype(np.float64)))
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'matthews_correlation_score': best_score}
    return search_result
