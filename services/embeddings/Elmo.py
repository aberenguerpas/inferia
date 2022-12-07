import tensorflow_hub as hub
import tensorflow.compat.v1 as tf
import numpy as np

class Elmo:

    def __init__(self):
       tf.compat.v1.disable_eager_execution()
       self.model = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
       self.dimensions = 768

    def getEmbedding(self, data):
       
        embeddings = self.model(data,
            signature="default",
            as_dict=True)["elmo"]
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
        
        return [np.mean(emb, axis=0) for emb in sess.run(embeddings)]