from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
import numpy as np

class milv:
    def __init__(self, host):

        self.host = host
        self.collections = []

        print('Connecting to Milvus server', host , end='...')

        try:
            connections.connect(
            
                uri='tcp://'+self.host+':19530'
            )
            print('Ok')
        except Exception as e:
            print('Error connecting to milvus server')
            print(e)


    def createCollection(self, collection_name, dim=768):

        if not utility.has_collection(collection_name):

            print('Creating a collection...')

            id = FieldSchema(
                name="id", 
                dtype=DataType.INT64,
                is_primary=True,
                auto_id = True
            )

            table_id = FieldSchema(
                name="table_id", 
                dtype=DataType.VARCHAR,
                max_length=500,
            )

            content = FieldSchema(
                name="data", 
                dtype=DataType.FLOAT_VECTOR, 
                dim=dim
            )

            data_desc = FieldSchema(
                name="data_desc", 
                dtype=DataType.VARCHAR, 
                max_length=1000,
            )

            schema = CollectionSchema(
                fields=[id, table_id, data_desc, content], 
                description="Collection of embeddings " + collection_name
            )

            collection = Collection(
                name=collection_name, 
                schema=schema, 
                using='default', 
                shards_num=2,
            )

            print("Created collection ", collection_name)

            self.collections.append(collection)


    def loadCollection(self, name = None):

        if name == None:
            for collection in self.collections:
                collection.load()
        else:
            collection = Collection(name)
            collection.load()

    def releaseCollection(self, name):

        if name == None:
            for collection in self.collections:
                collection.release()
        else:
            collection = Collection(name)
            collection.release()


    def collectionsProperties(self, name):
        collection = Collection(name)
        data = {}
        data['name'] = collection.name
        data['description'] = collection.description
        data['schema'] = collection.schema
        data['is_empty'] = collection.is_empty
        data['num_entities'] =  collection.num_entities

        print(data)
        return data

    def insertData(self, data, collection_name):
        
        collection = Collection(collection_name)      # Get an existing collection.
        collection.insert(data)

    def buildIndex(self, collection_name):
        print('Building index...',end='')
        index_params = {
            "metric_type":"IP",
            "index_type":"IVF_FLAT",
            "params":{
                "nlist": 1024
            }
        }

        collection = Collection(collection_name)      # Get an existing collection.
        collection.create_index(
            field_name="data", 
            index_params=index_params,
            index_name="index"
        )
        print('Ok')

            
    def closeConnection(self):
        connections.disconnect('default')

