# Model embeddings experiment

Prerequisites, install dependencies

-Create virtual env: 
```python -v venv env```

-Activate env: ```source env/bin/activate```

-Install requirements:
```pip install -r requirements.txt```

Steps:
1. Download and extract wikitables in ```experiments/data/wikitables``` using ```wget http://iai.group/downloads/smart_table/WP_tables.zip```

2. Execute `python services/extraction/extractor.py -i experiments/data/wikitables -o experiments/data/wikitables_clean`: extract all Wikitables in JSON format and generates an output folder with CSV version.

3. `python services/embeddings/main.py -m stb`: Run the embeddgins microservice

4. `python services/indexation/main.py -m stb`: Starts to index all tables in milvus database

5. `python services/search/search.py`: Perform the queries and store the result

6. Execute trec_eval from inferia directory:
```
trec_eval-9.0.7/trec_eval experiments/data/benchmarks/table/qrels.txt search_result/results.csv
```
