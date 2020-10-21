# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 11:57:54 2020

@author: TEJA
"""

    
from absl import logging

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import tensorflow_hub as hub
import sentencepiece as spm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns

module = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-lite/2")


def pre_process_sentences(messages):
    input_placeholder = tf.sparse_placeholder(tf.int64, shape=[None, None])
    encodings = module(
    inputs=dict(
        values=input_placeholder.values,
        indices=input_placeholder.indices,
        dense_shape=input_placeholder.dense_shape))

    with tf.Session() as sess:
        spm_path = sess.run(module(signature="spm_path"))

    sp = spm.SentencePieceProcessor()
    sp.Load(spm_path)
    print("SentencePiece model loaded at {}.".format(spm_path))
    
    def process_to_IDs_in_sparse_format(sp, sentences):
        ids = [sp.EncodeAsIds(x) for x in sentences]
        max_len = max(len(x) for x in ids)
        dense_shape=(len(ids), max_len)
        values=[item for sublist in ids for item in sublist]
        indices=[[row,col] for row in range(len(ids)) for col in range(len(ids[row]))]
        
        return (values, indices, dense_shape)
    
    values, indices, dense_shape = process_to_IDs_in_sparse_format(sp, messages)
    logging.set_verbosity(logging.ERROR)


    with tf.Session() as session:
      session.run([tf.global_variables_initializer(), tf.tables_initializer()])
      message_embeddings = session.run(encodings,feed_dict={input_placeholder.values: values,
                input_placeholder.indices: indices,
                input_placeholder.dense_shape: dense_shape})
    
    return message_embeddings



messages =list(df['title'])


    
message_embeddings=pre_process_sentences(messages)



np.shape(message_embeddings)




df = pd.read_csv("pre_processed.csv")



Settings = {
   "settings":{
      "number_of_shards":2,
      "number_of_replicas":1
   },
   "mappings":{
           "dynamic":"true",
           "_source":{
                "enabled":"true"},
       "properties":{
          "ml_vector":{
         "type":"dense_vector",
         "dims":512
      } 
    }
   }
}
    
          
ENDPOINT = "http://localhost:9200/"
es = Elasticsearch(hosts=ENDPOINT)


es.ping()
IndexName = 'netflix_ml'
my = es.indices.create(index=IndexName, ignore=[400,404],body=Settings)

my

df.columns

def generator(df2):
    for c, line in enumerate(df2):
        yield {
    '_index': 'netflix_ml',
    '_type': '_doc',
    '_id': c,
    '_source': {
        "title":line.get("title", ""),
       'director':line.get('director', ""),
        'description':line.get('description', ""),
        'ml_vector':line.get('ml_vector', "")
    }
        }
    raise StopIteration
    

df22 = df.to_dict('records')
df22
next(generator(df22))

try:
    res = helpers.bulk(es, generator(df22))
    print("Working")
except Exception as e:
    pass



input_query = "Krish Trish and Baltiboy: Best Friends Forever"

x=pre_process_sentences(input_query)

x=list(x[0])
x

len(x)

df['ml_vector']
            
script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, df['ml_vector']) + 1.0",
                "params": {"query_vector": x}
            }
        }
    }



response=es.search(
        index="netflix_ml",
        body={
            "size":10,
            "query":script_query,
            "_source":{"includes":["title","body"]}
            }
)


response


