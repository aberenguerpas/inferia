# InferIA experiment


Steps:

1. `services/extraction/extractor.py`: extract all Wikitables in JSON format and generates an output folder with CSV version.

2. `services/embeddings/main.py`: Run the microservice to 

3. `analysis.py`: get statistics of the similarity between CSV tables based on the results of the previous script
 
4. `coverage.py`: calculate statistics about the coverage of each model for the whole dataset, taking into account all the content, only string data and only numerical data.




Execute trec_eval
```
trec_eval-9.0.7/trec_eval experiments/data/benchmarks/table/qrels.txt search_result/results.csv
```
