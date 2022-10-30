import argparse
import glob
import os
import pandas as pd
import random
from tqdm import tqdm

def clean_text(text):
    """
    Given a text in this format "[Jan_Frans_van_Dael|Jan Frans van Dael]" return "Jan Frans van Dael".
    :param text: original text
    :return: cleaned text
    """
    cleaned = text
    if cleaned and isinstance(cleaned, str) and cleaned[0] == '[' and cleaned[-1] == ']' and '|' in cleaned:
        cleaned = cleaned.split('|')[1][:-1]

    return cleaned


def create_csv(input_dir, output_dir):
    """
    Reads JSON files from an input directory and writes them as CSV in an output directory.
    Selects only columns and rows content from the JSON.
    :param input_dir: input directory containing the JSON files
    :param output_dir: output directory where CSV files are written
    :return: none
    """
    list_files = glob.glob(os.path.join(input_dir, '*.json'))
    for file in tqdm(list_files):
        input_data = pd.read_json(file, orient='index')
        for _, table in input_data.iterrows():
            if table['numDataRows'] >= 10 and table['numCols'] > 0:  # Discard tables with less than 10 rows or no columns
                file_name = os.path.join(output_dir, table.name + '.csv')
                #print('Processing file "' + file_name + '"')
                # Columns 
                list_columns = list(map(clean_text, table['title']))  # Clean all the column names
                output_data = pd.DataFrame(columns=list_columns)
                # Rows
                list_rows = table['data']
                for row in list_rows:
                    row = list(map(clean_text, row))
                    output_data.loc[len(output_data)] = row
                output_data.to_csv(file_name, index=False)


def main():
    random.seed(777)  # Make it reproducible
    # Command line arguments
    parser = argparse.ArgumentParser(description='Process WikiTables corpus.')
    parser.add_argument('-i', '--input', required=True,
                        help='Name of the input folder storing the WikiTables files in JSON format')
    parser.add_argument('-o', '--output', required=True,
                        help='Name of the output folder to store the resulting CSV files')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    # Create CSV files from JSON files
    create_csv(args.input, args.output)


if __name__ == '__main__':
    main()