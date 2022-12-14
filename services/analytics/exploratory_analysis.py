import argparse
import glob
import pandas as pd
import os
from tqdm import tqdm

def main():
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    parser = argparse.ArgumentParser(description='Corpus Exploratory Analysis')
    parser.add_argument('-i', '--input', required=True,
                        help='Name of the input folder storing the corpus CSV files ')
    args = parser.parse_args()

    list_files = glob.glob(os.path.join(args.input, '*.csv'))

    data = []
    for file in tqdm(list_files):
        df = pd.read_csv(file, encoding = 'ISO-8859-1', on_bad_lines='skip')

        data.append([len(df.index),len(df.columns)])
        #table = pd.concat([table, new_table], axis=0)  # Add the rows of new the new table to the previous one
 
    df_result = pd.DataFrame(data=data ,columns=['n_rows', 'n_columns'])

    pd.set_option('display.max_columns', None)  # Show all the column values
    print('All:')
    print(df_result.describe())
    print('Mean:')
    print( df_result.mean())

    ax = df_result.plot.hist(bins=12, alpha=0.5)
    fig = ax.get_figure()
    fig.savefig('histogram.pdf')

if __name__ == '__main__':
    main()