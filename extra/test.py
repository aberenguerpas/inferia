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

tables_q = pd.read_csv("experiments/data/benchmarks/table/queries.txt", sep="\t", header=None)
tables_q_id = tables_q.iloc[:,1].tolist()

tables_a = pd.read_csv("experiments/data/benchmarks/table/qrels.txt", sep="\t", header=None)
tables_a_id = tables_a.iloc[:,2].tolist()

tables_e = list(dict.fromkeys(tables_q_id + tables_a_id))


tables_f = os.listdir("experiments/data/wikitables_clean")

tables_f = list(map(lambda table: table.split(".")[0], tables_f))

tables_common = list(common_member(tables_e, tables_f))


tables_q = tables_q[tables_q.iloc[:,1].isin(tables_common)] # Solo las queries que no han sido filtradas

ids_filtrados = tables_q.iloc[:,0].tolist()

tables_a = tables_a[tables_a.iloc[:,0].isin(ids_filtrados)]
tables_a = tables_a[tables_a.iloc[:,2].isin(tables_common)]

tables_q.to_csv('experiments/data/benchmarks/table/queries_r.txt',sep='\t', header=False, index=False)
tables_a.to_csv('experiments/data/benchmarks/table/qrels_r.txt',sep='\t', header=False, index=False)
