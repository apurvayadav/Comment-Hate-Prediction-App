# Importing required libraries
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import gradio as gr
from tensorflow.keras.layers import TextVectorization

# Importing Data
data = pd.read_csv("train.csv")

# Creating Word Embeddings

X = data['comment_text']
y = data[data.columns[2:]].values
MAX_FEATURES = 200000
vectorizer = TextVectorization(max_tokens = MAX_FEATURES, output_sequence_length = 1800, output_mode = 'int')
vectorizer.adapt(X.values)
vectorized_text = vectorizer(X.values)
print('Vectorization Complete!')

# Loading The Model
model = tf.keras.models.load_model('hate_model.h5')

# To display results
def predict_comment_hate(comment):
    comment_vectorized = vectorizer(comment)
    results = model.predict(np.expand_dims(comment_vectorized,0))
    
    text = ''
    for idx, col in enumerate(data.columns[2:]):
        text += '{}: {}\n'.format(col, results[0][idx]>0.5)
        
    return text

interface = gr.Interface(fn= predict_comment_hate, inputs= gr.inputs.Textbox(lines= 2, placeholder= 'Enter the Comment'), outputs = 'text')
interface.launch()
