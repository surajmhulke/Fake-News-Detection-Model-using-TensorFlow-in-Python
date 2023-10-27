# Fake-News-Detection-Model-using-TensorFlow-in-Python

Fake News Detection Model using TensorFlow in Python
 
Fake News means incorporating information that leads people to the wrong paths. It can have real-world adverse effects that aim to intentionally deceive, gain attention, manipulate public opinion, or damage reputation. It is necessary to detect fake news mainly for media outlets to have the ability to attract viewers to their website to generate online advertising revenue.

#  Fake News Detection Model using TensorFlow in Python
In this article, we are going to develop a Deep learning model using Tensorflow and use this model to detect whether the news is fake or not.

We will be using fake_news_dataset, which contains News text and corresponding label (FAKE or REAL). Dataset can be downloaded from this link.

The steps to be followed are : 


Importing Libraries and dataset
Preprocessing Dataset
Generating Word Embeddings
Model Architecture
Model Evaluation and Prediction
#  Importing Libraries and Dataset
The libraries we will be using are :

NumPy: To perform different mathematical functions. 
Pandas: To load dataset.
Tensorflow: To preprocessing the data and to create the model.
SkLearn: For train-test split and to import the modules for model evaluation.
import numpy as np 
import pandas as pd 
import json 
import csv 
import random 
  
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras import regularizers 
  
import pprint 
import tensorflow.compat.v1 as tf 
from tensorflow.python.framework import ops 
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing 
tf.disable_eager_execution() 
  
# Reading the data 
data = pd.read_csv("news.csv") 
data.head() 
 
 

#  Preprocessing Dataset
As we can see the dataset contains one unnamed column. So we drop that column from the dataset.

data = data.drop(["Unnamed: 0"], axis=1) 
data.head(5) 
 

 
 

# Data Encoding
It converts the categorical column (label in out case) into numerical values.

# encoding the labels 
le = preprocessing.LabelEncoder() 
le.fit(data['label']) 
data['label'] = le.transform(data['label']) 
These are some variables required for the model training.

embedding_dim = 50
max_length = 54
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 3000
test_portion = .1
# Tokenization 
This process divides a large piece of continuous text into distinct units or tokens basically. Here we use columns separately for a temporal basis as a pipeline just for good accuracy.

title = [] 
text = [] 
labels = [] 
for x in range(training_size): 
    title.append(data['title'][x]) 
    text.append(data['text'][x]) 
    labels.append(data['label'][x]) 
# Applying Tokenization

tokenizer1 = Tokenizer() 
tokenizer1.fit_on_texts(title) 
word_index1 = tokenizer1.word_index 
vocab_size1 = len(word_index1) 
sequences1 = tokenizer1.texts_to_sequences(title) 
padded1 = pad_sequences( 
    sequences1,  padding=padding_type, truncating=trunc_type) 
split = int(test_portion * training_size) 
training_sequences1 = padded1[split:training_size] 
test_sequences1 = padded1[0:split] 
test_labels = labels[0:split] 
training_labels = labels[split:training_size] 
# Generating Word Embedding
It allows words with similar meanings to have a similar representation. Here each individual word is represented as real-valued vectors in a predefined vector space. For that we will use glove.6B.50d.txt. It has the predefined vector space for words. You can download the file using this link.

embeddings_index = {} 
with open('glove.6B.50d.txt') as f: 
    for line in f: 
        values = line.split() 
        word = values[0] 
        coefs = np.asarray(values[1:], dtype='float32') 
        embeddings_index[word] = coefs 
  
# Generating embeddings 
embeddings_matrix = np.zeros((vocab_size1+1, embedding_dim)) 
for word, i in word_index1.items(): 
    embedding_vector = embeddings_index.get(word) 
    if embedding_vector is not None: 
        embeddings_matrix[i] = embedding_vector 
#  Creating Model Architecture
Now it’s time to introduce TensorFlow to create the model.  Here we use the TensorFlow embedding technique with Keras Embedding Layer where we map original input data into some set of real-valued dimensions.

model = tf.keras.Sequential([ 
    tf.keras.layers.Embedding(vocab_size1+1, embedding_dim, 
                              input_length=max_length, weights=[ 
                                  embeddings_matrix], 
                              trainable=False), 
    tf.keras.layers.Dropout(0.2), 
    tf.keras.layers.Conv1D(64, 5, activation='relu'), 
    tf.keras.layers.MaxPooling1D(pool_size=4), 
    tf.keras.layers.LSTM(64), 
    tf.keras.layers.Dense(1, activation='sigmoid') 
]) 
model.compile(loss='binary_crossentropy', 
              optimizer='adam', metrics=['accuracy']) 
model.summary() 
 
 

num_epochs = 50
  
training_padded = np.array(training_sequences1) 
training_labels = np.array(training_labels) 
testing_padded = np.array(test_sequences1) 
testing_labels = np.array(test_labels) 
  
history = model.fit(training_padded, training_labels,  
                    epochs=num_epochs, 
                    validation_data=(testing_padded, 
                                     testing_labels),  
                    verbose=2) 
 

Model Evaluation and Prediction
Now, the detection model is built using TensorFlow. Now we will try to test the model by using some news text by predicting whether it is true or false.

# sample text to check if fake or not 
X = "Karry to go to France in gesture of sympathy"
  
# detection 
sequences = tokenizer1.texts_to_sequences([X])[0] 
sequences = pad_sequences([sequences], maxlen=54, 
                          padding=padding_type,  
                          truncating=trunc_type) 
if(model.predict(sequences, verbose=0)[0][0] >= 0.5): 
    print("This news is True") 
else: 
    print("This news is false") 
Output : 

This news is false
Conclusion
In this way, we can build a fake news detection model using TensorFlow using python.
