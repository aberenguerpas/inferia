import os
from utils import *
import numpy as np
from tqdm import tqdm
import warnings
import time
import argparse
import pandas as pd
import traceback
import re
from scipy import spatial


def get_column_text(column):
    """
    Get the word embedding vector that represents all the text contained in a column.
    :param column: content of the column
    :param model: pre-trained model loaded in memory
    :param model_name: name of the model passed
    :return: word embedding vector in the form of a list of float values
    """
    column.dropna(inplace=True)  # Remove NaN values from the column
    if column.size > 0:
        if column.dtype != object or isinstance(column.iloc[0], bool):
            column = column.apply(str)  # Convert numbers and booleans to string

        # Transform to string in case Pandas assigns "object" type to numeric columns and "join" fails (sometimes it happens)
        try:
            # Lowercase the text because the sentence model is uncased
            text = ' '.join(column.to_list()).lower()
        except TypeError:
            text = ' '.join(column.apply(str).to_list()).lower()

        # The maximum token length admitted by is 256
        max_sequence_length = 256
        # Larger texts are cut into 256 token length pieces and their vectors are averaged
        # Take into account that a cell could have more than one token
        text = re.sub(r'[^\w\s]', '', text)
        if len(text.split()) > max_sequence_length:

            list_tokens = text.split()
           
            list_texts = []
            for i in range(0, column.size, max_sequence_length):
                
                list_texts.append(' '.join(list_tokens[i:i+max_sequence_length]))
            
            return list_texts
        else:
            return [text]
    else:
        return []


def compare_tables(vector_table, subtable):
    """
    Compute the similarity between two vectors as the average similarity of their columns.
    :param table: original table
    :param subtable: table containing a subset of rows of the original table
    :param model: pre-trained model loaded in memory
    :param model_name: name of the pre-trained model passed
    :return: value between 0 and 1 indicating the similarity between the original and the subset table
    """

    list_similarity = []

    vectors_subtable = []
    for i, column in enumerate(subtable):

        column_text = get_column_text(subtable[column])
       

        if column_text:
            if len(column_text)>50:
                aux = []
                for i in range(0, len(column_text), 50):
                    if column_text[i:50]:
                        aux.append(getEmbeddings(column_text[i:50]))

                vectors_subtable.append(aux)
            else:
               
                vectors_subtable.append(getEmbeddings(column_text))
        else:
            vectors_subtable.append([])

    for i, column in enumerate(subtable):

        n_vectors_subtable = len(vectors_subtable[i]) 

        if n_vectors_subtable > 1:
            vector_subtable = np.mean(vectors_subtable[i], axis=0).tolist()
            
        elif n_vectors_subtable == 1:
            vector_subtable = vectors_subtable[i][0]
        else:
            vector_subtable = []

        # Compare each column from the table with the corresponding column in the subtable
        if len(vector_table[column])==50:
            print(vector_table[column])
        if vector_table[column] and vector_subtable:  # Discard empty lists

            output = 1 - spatial.distance.cosine(vector_table[column], vector_subtable)
            
            list_similarity.append(output)

    return sum(list_similarity)/len(subtable.columns)  # Calculate the mean (empty columns penalize)

def calculate_similarity(table, random_state):
    """
    Given a table, extract a subset of tables of different sizes and compare their similarity with the original.
    :param table: original table
    :param model: pre-trained model loaded in memory
    :param model_name: name of the pre-trained model passed
    :param random_state: seed for random generator
    :return: list with the similarity between tables for each subset
    """

    # Calculamos los vectores de la tabla para que no se tenga que repetir en cada iteracion
    vectors_table = []
    vector_table = dict()
    for i, column in enumerate(table):
        column_text = get_column_text(table[column])

        if column_text:
            if len(column_text)>50:
                aux = []
                for i in range(0, len(column_text), 50):
                    if column_text[i:50]:
                        aux.append(getEmbeddings(column_text[i:50]))

                vectors_table.append(aux)
            else:
                vectors_table.append(getEmbeddings(column_text))
        else:
            vectors_table.append([])


    for i, column in enumerate(table):

        n_vectors_table = len(vectors_table[i]) 

        if n_vectors_table > 1:
            vector_table[column] = np.mean(vectors_table[i], axis=0).tolist()
        elif  n_vectors_table == 1:
            vector_table[column] = vectors_table[i][0]
        else:
            vector_table[column] = []

    # Comparacion
    list_similarity = [np.NaN] * 11  # Initialize 11 values to NaN: 1%, 5%, 10%, ..., 90%
    table_size = len(table.index)
    if table_size >= 100:  # Calculate similarity with 1% of the rows
        subtable = table.sample(n=round(table_size * 0.01), random_state=random_state).sort_index()
        list_similarity[0] = compare_tables(vector_table, subtable)
    if table_size >= 20:  # Calculate similarity with 5% of the rows
        subtable = table.sample(n=round(table_size * 0.05), random_state=random_state).sort_index()
        list_similarity[1] = compare_tables(vector_table, subtable)

    
    index = 2  # Controls where to store the values in the list of similarity values
    for percentage in range(10, 90 + 1, 10):  # Starting from 10% until 90% in 10% steps
        # Extract a subset of percentage rows ('sort_index()' keeps row order)
        subtable = table.sample(n=round(table_size * percentage / 100), random_state=random_state).sort_index()
        list_similarity[index] = compare_tables(vector_table, subtable)
        index += 1
    return list_similarity


def main():

    start_time = time.time()
    parser = argparse.ArgumentParser(description='Process WikiTables corpus')
    parser.add_argument('-i', '--input', default='experiments/data/wikitables_clean', help='Name of the input folder storing CSV tables')
    parser.add_argument('-m', '--model', default='brt',
                        help='Model to use: "sbt" (Sentence-BERT, Default), "rbt" (Roberta),"fst" (fastText), "w2v"(Word2Vec)) '
                             ' "brt" (Bert)')
    parser.add_argument('-r', '--result', default='./result',
                        help='Name of the output folder that stores the similarity values calculated')
    parser.add_argument('-s', '--subset', default='all', choices=['all', 'str', 'num'],
                        help='Use all ("all"), string ("str") or numeric ("num") columns')
    parser.add_argument('-R', '--resume', action='store_true', help='Resume the generate embedding process in case something failed.')
    parser.add_argument('-rs', '--rstate', default=None, type=int, help='Seed value for random selection of rows')
    args = parser.parse_args()

    # Config
    np.set_printoptions(suppress=True)
    warnings.filterwarnings("ignore")

    # Create the output directory if it does not exist
    if not os.path.exists(args.result+"_"+args.model):
        os.makedirs(args.result+"_"+args.model)

    # Store the similarity of each table with the subsets 1%, 5%, 10%, 20%, ..., 90%
    data_similarity = pd.DataFrame(columns=['Name', '1%', '5%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%'])
    data_similarity.set_index('Name', inplace=True)  # The name of the table is the index of the DataFrame
    tables_discarded = 0

    # Index data
    number_files = len(os.listdir(args.input))
    start = 1
    for i, path in enumerate(tqdm(os.listdir(args.input))):
        file = open(os.path.join(args.input, path))

        try:
            table = pd.read_csv(file, encoding = 'ISO-8859-1', on_bad_lines='skip')
            
            table.dropna(axis=1, how='all', inplace=True)  # Remove columns where all elements are NaN
            table.dropna(how='all', inplace=True)  # Remove rows where all elements are NaN

            if len(table.index) >= 10:  # Discard tables with less than 10 rows after dropping NaN
                data_similarity.loc[os.path.basename(path).split('.')[0]] = calculate_similarity(table, args.rstate)
            else:
                tables_discarded += 1

            # Save data every 10,000 files or after processing the last file
            if (i+1) % 10 == 0 or i == number_files-1:
                end = i+1
                data_similarity.to_csv(os.path.join(args.result+"_"+args.model, args.model + '_' + str(start) + '-' + str(end) + '.csv'))
                data_similarity = data_similarity[0:0]  # Erase the rows and keep the same DataFrame structure (columns)
                start = end+1

        except Exception as e:
            print(e)
            print(os.path.basename(path).split('.')[0])
            traceback.print_exc()

    print('Total tables discarded: ' + str(tables_discarded))

    end_time = time.time()
    print('Processing time: ' + str(end_time - start_time) + ' seconds')


if __name__ == "__main__":
    main()
