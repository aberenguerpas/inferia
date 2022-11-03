from milv import milv
import os
import torch
from utils import *
import numpy as np
from tqdm import tqdm
import warnings
import time
import argparse
import pandas as pd
import traceback


def get_column_vector(column):
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

        if len(text.split()) > max_sequence_length:

            list_tokens = text.split()
           
            list_vectors = []
            for i in range(0, column.size, max_sequence_length):
                
                list_vectors.append(getEmbeddings(' '.join(list_tokens[i:i+max_sequence_length])))
            
            return np.mean(list_vectors, axis=0).tolist()
        else:
            return getEmbeddings(text)
    else:
        return []


def index_table(table, model_name, key):
    for column in table:
        try:
            vector_col = get_column_vector(table[column])
            col_emb = [[key], ['Col '+ column[:50]], [normalize(np.array(vector_col))]]
            milvus.insertData(col_emb, model_name+"_content_100")
        except Exception as e:
            print(e)
            traceback.print_exc()

def compare_tables(table, subtable):
    """
    Compute the similarity between two vectors as the average similarity of their columns.
    :param table: original table
    :param subtable: table containing a subset of rows of the original table
    :param model: pre-trained model loaded in memory
    :param model_name: name of the pre-trained model passed
    :return: value between 0 and 1 indicating the similarity between the original and the subset table
    """
    list_similarity = []
    for column in table:
        vector_table = get_column_vector(table[column])
        #col_emb = [[key], ['Col'+ col], [normalize(np.array(emb))]]
        #milvus.insertData(col_emb, model_name+"_content_100")

        vector_subtable = get_column_vector(subtable[column])

        # Compare each column from the table with the corresponding column in the subtable
        if vector_table and vector_subtable:  # Discard empty lists
            cos = torch.nn.CosineSimilarity(dim=0)
            output = cos(vector_table, vector_subtable)
            list_similarity.append(output)

    return sum(list_similarity)/len(table.columns)  # Calculate the mean (empty columns penalize)

def calculate_similarity(table, model_name, random_state):
    """
    Given a table, extract a subset of tables of different sizes and compare their similarity with the original.
    :param table: original table
    :param model: pre-trained model loaded in memory
    :param model_name: name of the pre-trained model passed
    :param random_state: seed for random generator
    :return: list with the similarity between tables for each subset
    """
    list_similarity = [np.NaN] * 11  # Initialize 11 values to NaN: 1%, 5%, 10%, ..., 90%
    table_size = len(table.index)
    if table_size >= 100:  # Calculate similarity with 1% of the rows
        subtable = table.sample(n=round(table_size * 0.01), random_state=random_state).sort_index()
        list_similarity[0] = compare_tables(table, subtable,  model_name)
    if table_size >= 20:  # Calculate similarity with 5% of the rows
        subtable = table.sample(n=round(table_size * 0.05), random_state=random_state).sort_index()
        list_similarity[1] = compare_tables(table, subtable, model_name)

    index = 2  # Controls where to store the values in the list of similarity values
    for percentage in range(10, 90 + 1, 10):  # Starting from 10% until 90% in 10% steps
        # Extract a subset of percentage rows ('sort_index()' keeps row order)
        subtable = table.sample(n=round(table_size * percentage / 100), random_state=random_state).sort_index()
        list_similarity[index] = compare_tables(table, subtable, model_name)
        index += 1

    return list_similarity


def main():

    start_time = time.time()
    parser = argparse.ArgumentParser(description='Process WikiTables corpus')
    parser.add_argument('-i', '--input', default='experiments/data/wikitables_clean', help='Name of the input folder storing CSV tables')
    parser.add_argument('-m', '--model', default='brt', choices=['stb', 'rbt', 'brt'],
                        help='Model to use: "stb" (Sentence-BERT, Default), "rbt" (Roberta),'
                             ' "brt" (Bert)')
    parser.add_argument('-r', '--result', default='./result',
                        help='Name of the output folder that stores the similarity values calculated')
    parser.add_argument('-s', '--subset', default='all', choices=['all', 'str', 'num'],
                        help='Use all ("all"), string ("str") or numeric ("num") columns')
    parser.add_argument('-R', '--resume', action='store_true', help='Resume the generate embedding process in case something failed.')
    parser.add_argument('-rs', '--rstate', default=None, type=int, help='Seed value for random selection of rows')
    parser.add_argument('-t', '--type', default='all', choices=['all', 'split'], help='Experiment type "all" to index full table or "split" to index subtables 1,5,10,20%...90%')
    args = parser.parse_args()

    # Config
    np.set_printoptions(suppress=True)
    warnings.filterwarnings("ignore")

    # Create the output directory if it does not exist
    if not os.path.exists(args.result):
        os.makedirs(args.result)

    # model
    model_name = args.model
    # start microservice embeddings
    #Popen(['env/bin/python','services/embeddings/main.py', '-m', args.model], stdin=PIPE, stdout=PIPE)
    #time.sleep(10)

    if model_name == 'stb':
        dimensions = 384
    else:
        dimensions = 768

    # Store the similarity of each table with the subsets 1%, 5%, 10%, 20%, ..., 90%
    if args.type == 'split':
        data_similarity = pd.DataFrame(columns=['Name', '1%', '5%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
        data_similarity.set_index('Name', inplace=True)  # The name of the table is the index of the DataFrame
        milvus.createCollection(model_name+"_content_1", dim = dimensions)
        milvus.createCollection(model_name+"_content_5", dim = dimensions)
        for i in range(10,110,10):
            milvus.createCollection(model_name+"_content_"+str(i), dim = dimensions)
    else:
        # Create collections
        milvus.createCollection(model_name+"_headers", dim = dimensions)
        milvus.createCollection(model_name+"_content_100", dim = dimensions)
        # Load collections
        milvus.loadCollection(model_name+"_headers")
        milvus.loadCollection(model_name+"_content_100")

    tables_discarded = 0
    #list_files = glob.glob(os.path.join(args.input, '*.csv'))

    milvus.createCollection(model_name+"_headers", dim = dimensions)
    milvus.loadCollection()

    # Index data
    number_files = len(os.listdir(args.input))
    start = 1
    for i, path in enumerate(tqdm(os.listdir(args.input))):
        file = open(os.path.join(args.input, path))

        try:
            key = path.split('.')[0] # Milvus id
            table = pd.read_csv(file)
            table.dropna(axis=1, how='all', inplace=True)  # Remove columns where all elements are NaN
            table.dropna(how='all', inplace=True)  # Remove rows where all elements are NaN
            if len(table.index) >= 1:  # Discard tables with less than 10 rows after dropping NaN
                if args.type == 'all':
                    # index table head embedding 
                    headers = table.columns.values
                    headers = filter(lambda col: 'Unnamed' not in col, headers) # Skip unnamed column
                    headers_text = ' '.join(map(str, headers))

                    head_emb = [[key],['headers'], [normalize(np.array(getEmbeddings(headers_text)))]]
                    milvus.insertData(head_emb, model_name+"_headers")
                    index_table(table, args.model, key)
                else:
                    data_similarity.loc[os.path.basename(file).split('.')[0]] = calculate_similarity(table, args.model, args.rstate)

            else:
                tables_discarded += 1

            # Save data every 10,000 files or after processing the last file
            if (i+1) % 10000 == 0 or i == number_files-1:
                end = i+1
                if args.type == 'split':
                    data_similarity.to_csv(os.path.join(args.result, args.model + '_' + str(start) + '-' + str(end) + '.csv'))
                    data_similarity = data_similarity[0:0]  # Erase the rows and keep the same DataFrame structure (columns)
                start = end+1

                # Build indexs
                if args.type == 'split':
                    milvus.buildIndex(model_name+"_headers")
                    milvus.buildIndex(model_name+"_content_1")
                    milvus.buildIndex(model_name+"_content_5")
                    for i in range(10,110,10):
                        milvus.buildIndex(model_name+"_content_"+str(i))
                else:
                    milvus.buildIndex(model_name+"_headers")
                    milvus.buildIndex(model_name+"_content_100")

        except Exception as e:
            print(e)
            print(path)
            traceback.print_exc()

    print('Total tables discarded: ' + str(tables_discarded))

    end_time = time.time()
    print('Processing time: ' + str(end_time - start_time) + ' seconds')

    milvus.closeConnection()

if __name__ == "__main__":
    milvus = milv('146.59.196.180')
    main()
