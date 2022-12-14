import argparse
import glob
import pandas as pd
import os


def main():
    parser = argparse.ArgumentParser(description='Analyse table similarity files.')
    parser.add_argument('-i', '--input', required=True,
                        help='Name of the input folder storing the CSV files with similarity values of the tables')
    args = parser.parse_args()

    list_files = glob.glob(os.path.join(args.input, '*.csv'))
    try:
        table = pd.read_csv(list_files.pop(0), index_col='Name')  # Store the first file
        for file in list_files:
            new_table = pd.read_csv(file, index_col='Name')
            table = pd.concat([table, new_table], axis=0)  # Add the rows of new the new table to the previous one
        # table.to_csv('salida.csv')
        pd.set_option('display.max_columns', None)  # Show all the column values
        print('All:')
        print(table.describe())
        print('Mean:')
        print(table.mean())
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()