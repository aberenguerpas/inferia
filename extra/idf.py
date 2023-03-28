import os
from tqdm import tqdm
import pandas as pd
import pickle


def add_dic(t):
    if t in f_i.keys():
        f_i[t] += 1
    else:
        f_i[t] = 1


PATH_DATA = '/home/wake/inferia/experiments/data/dublin_clean'
total_columns = 0


f_i = dict()
for path in tqdm(os.listdir(PATH_DATA)):
    file = os.path.join(PATH_DATA, path)

    table = pd.read_csv(file)
    table.dropna(axis=1,
                 how='all',
                 inplace=True)  # Remove columns where all elements are NaN
    table.dropna(how='all',
                 inplace=True)  # Remove rows where all elements are NaN

    total_columns += len(table.columns)

    # Preprocess column

    for c in table.columns:
        col_values = table[c].unique()
        col_values = [add_dic(str(item).lower()) for item in col_values]

with open('f_i.pickle', 'wb') as handle:
    pickle.dump(f_i, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('total_cols.txt', 'w') as f:
    f.write(str(total_columns))
