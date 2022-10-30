"""
This file pretends to check the tables % available after the filter 
"""

import pandas as pd
import os

def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
 
    if (a_set & b_set):
        return a_set & b_set
    else:
        print("No common elements")


# Read the queries.txt tables

tables_q = pd.read_csv("experiments/data/benchmarks/table/queries.txt", sep="\t")
tables_q = tables_q.iloc[:,1].tolist()

tables_a = pd.read_csv("experiments/data/benchmarks/table/qrels.txt", sep="\t")
tables_a = tables_a.iloc[:,2].tolist()

tables_e = list(dict.fromkeys(tables_q + tables_a))

print("Total tables used in search experiment:", len(tables_e))

# Available tables after filter

tables_f = os.listdir("experiments/data/wikitables_clean")
print("Total wikitables after filter:", len(tables_f))
tables_f = list(map(lambda table: table.split(".")[0], tables_f))

tables_common = len(list(common_member(tables_e, tables_f)))

print("Total common tables:", tables_common,"-" , str(round(tables_common/len(tables_e) * 100, 2)) + "%")



