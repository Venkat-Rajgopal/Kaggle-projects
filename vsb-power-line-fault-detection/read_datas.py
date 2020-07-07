import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from utilities import *

# change data dir based on machine
local = True
if local is True:
    data_dir = 'data/'
else:
    data_dir = '../input/vsb-power-line-fault-detection/'

df_train = pd.read_csv(data_dir + 'metadata_train.csv')
df_train = df_train.set_index(['id_measurement', 'phase'])


def prep_data(start, end):
    praq_train = pq.read_pandas(data_dir + 'train.parquet', columns=[str(i) for i in range(start, end)]).to_pandas()

    X = []
    y = []

    for id_measurement in tqdm(df_train.index.levels[0].unique()[int(start / 3):int(end / 3)]):
        x_signal = []
        # for each phase of the signal
        for phase in [0, 1, 2]:

            # extract from df_train both signal_id and target to compose the new data sets
            signal_id, target = df_train.loc[id_measurement].loc[phase]

            # but just append the target one time, to not triplicate it
            if phase == 0:
                y.append(target)
            x_signal.append(transform_ts(praq_train[str(signal_id)]))

        # concatenate all the 3 phases in one matrix
        x_signal = np.concatenate(x_signal, axis=1)
        X.append(x_signal)

    X = np.asarray(X)
    y = np.asarray(y)
    return X, y


X = []
y = []


def load_all():
    total_size = len(df_train)
    for ini, end in [(0, int(total_size / 2)), (int(total_size / 2), total_size)]:
        x_temp, y_temp = prep_data(ini, end)
        X.append(x_temp)
        y.append(y_temp)

# load_all()
# X = np.concatenate(X)
# y = np.concatenate(y)
#
# np.save(data_dir + "X.npy", X)
# np.save(data_dir + "y.npy", y)

# Preparing test side
print('Preparing Test set from metadata...')

meta_test = pd.read_csv(data_dir + 'metadata_test.csv')
meta_test = meta_test.set_index(['signal_id'])

first_sig = meta_test.index[0]
n_parts = 10
max_line = len(meta_test)
part_size = int(max_line / n_parts)
last_part = max_line % n_parts
print(first_sig, n_parts, max_line, part_size, last_part, n_parts * part_size + last_part)

# Here we create a list of lists with start index and end index for each of the 10 parts and one for the last partial part

start_end = [[x, x+part_size] for x in range(first_sig, max_line + first_sig, part_size)]
start_end = start_end[:-1] + [[start_end[-1][0], start_end[-1][0] + last_part]]
print(start_end)


x_test = []
# transforming the 3 phases 800000 measurement in matrix (160,57)
for start, end in start_end:
    subset_test = pq.read_pandas(data_dir + 'test.parquet', columns=[str(i) for i in range(start, end)]).to_pandas()
    for i in tqdm(subset_test.columns):
        id_measurement, phase = meta_test.loc[int(i)]
        subset_test_col = subset_test[i]
        subset_trans = transform_ts(subset_test_col)
        x_test.append([i, id_measurement, phase, subset_trans])


x_test_input = np.asarray([np.concatenate([x_test[i][3], x_test[i+1][3], x_test[i+2][3]], axis=1) for i in range(0, len(x_test), 3)])
np.save("data/x_test.npy", x_test_input)

print(x_test_input.shape)
