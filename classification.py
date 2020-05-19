#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import testsets
import evaluation
import pandas as pd
import numpy as np
import re
import nltk
import os
import pickle
import keras
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from keras import preprocessing, optimizers
from keras.callbacks import EarlyStopping
from nltk import word_tokenize
from numpy import argmax
from sklearn import metrics, svm
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, LSTM, Dropout, SpatialDropout1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def dataset(filename):
    data=[]
    contentlist = []
    classlist = []
    idlist = []
    with open(filename,'r',encoding='utf-8') as f:
        i = 0
        for line in f.readlines():
            linedata = {}
            linelist = line.split('\t')        
            linedata = {'id':linelist[0],'sentiment':linelist[1],'tweet':preprocess(linelist[2].lower())}
            data.append(linedata)
    for i in range(len(data)):
        contentlist.append(data[i]['tweet'])
        classlist.append(data[i]['sentiment'])
        idlist.append(data[i]['id'])
    return idlist,classlist,contentlist

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
def preprocess(sentence):
    content = ''
    sentence = sentence.lower()
    #Change links to 'URL'
    sentence = re.sub(r'https?://[\S]+', 'URL', sentence)
    #Change # and @ to 'HASH' and 'HNDL'
    sentence = re.sub(r'#(\w+)', r'hash_\1', sentence)
    sentence = re.sub(r'@(\w+)', r'hndl_\1', sentence)
    #Replace characters repeating more than twice as two characters
    sentence = re.sub(r'(.)\1{1,}', r'\1\1', sentence)
    #Remove all non-alphanumeric characters except spaces    
    sentence = re.sub(r'[^a-z0-9 ]+', '', sentence)
    #Remove words with only 1 character.
    sentence = re.sub(r'\b[a-z]\b', '',sentence)
    #Remove numbers that are fully made of digits
    sentence = re.sub(r'\b[0-9]+\b', '',sentence)
    # Convert to tokens  
    tokens = nltk.word_tokenize(sentence)
    tag = nltk.pos_tag(tokens)
    wnl = nltk.WordNetLemmatizer()
    for j in tag:
        lemma = wnl.lemmatize(j[0],get_wordnet_pos(j[1]))
        content += lemma+ ' '
    return content

id_train,sentiment_train,tweet_train = dataset('twitter-training-data.txt')

for classifier in ['naive Bayes','SVM','LSTM']: 
# You may rename the names of the classifiers to something more descriptive
    if classifier == 'naive Bayes':
        if os.path.exists('naiveBayes.pickle'):
            with open('naiveBayes.pickle','rb') as f:
                clf = pickle.load(f)
        else:
            
            print('Training ' + classifier)
            text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                                 ('tfidf', TfidfTransformer()),
                                 ('clf', MultinomialNB()),])
            parameters = {
                    'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
                    'tfidf__use_idf': (True, False),
                    'clf__alpha': (1, 1e-1, 1e-2, 1e-3),
            }
            gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
            clf = gs_clf.fit(tweet_train, sentiment_train)
            for para_name in sorted(parameters.keys()):
                print("%s:%r"%(para_name,gs_clf.best_params_[para_name]))
            with open('naiveBayes.pickle','wb') as f:
                pickle.dump(clf,f)
    elif classifier == 'SVM':
        if os.path.exists('SVM.pickle'):
            with open('SVM.pickle','rb') as f:
                clf = pickle.load(f)
        else:        
            print('Training ' + classifier)
            text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                                 ('tfidf', TfidfTransformer()),
                                 ('clf', SGDClassifier())
                                ])
            parameters = {
                    'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
                    'tfidf__use_idf': (True, False),
                    'clf__alpha': (1e-2, 1e-3),
                    'clf__penalty': ('l1', 'l2'),
                }
            gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
            clf = gs_clf.fit(tweet_train, sentiment_train)
            for para_name in sorted(parameters.keys()):
                print("%s:%r"%(para_name,gs_clf.best_params_[para_name]))
            with open('SVM.pickle','wb') as f:
                pickle.dump(clf,f)
    elif classifier == 'LSTM':
        if os.path.exists('LSTM.pickle'):
            with open('LSTM.pickle','rb') as f:
                clf = pickle.load(f)
        else:            
            embedding ={}
            f = open('glove.6B.100d.txt','r',encoding='utf-8')
            for line in f:
                value = line.split()
                embedding[value[0]]=np.asarray(value[1:],dtype='float32')
            f.close()
            max_words = 5000
            dim = 100
            tokenizer = Tokenizer(num_words = 5000)
            tokenizer.fit_on_texts(tweet_train)
            sequences = tokenizer.texts_to_sequences(tweet_train)
            max_len = min(len(tokenizer.word_index),5000)
            embedding_matrix = np.zeros((5000+1,100))
            for word,i in tokenizer.word_index.items():
                if i > 5000:
                    continue
                if word in embedding:
                    embedding_matrix[i]=embedding[word]
            tweet_train = pad_sequences(sequences, maxlen=dim)
            label_train = []
            for i in sentiment_train:
                if i == 'negative':
                    i = 0
                elif i == 'neutral':
                    i = 1
                else :
                    i = 2
                label_train.append(i)
            label_train = keras.utils.to_categorical(label_train, num_classes=3)
            lstm = Sequential()
            lstm.add(Embedding(5000+1, 100, input_length=100, weights=[embedding_matrix], trainable=False))
            lstm.add(LSTM(50))
            lstm.add(Dense(3, activation='softmax'))
            lstm.summary()
            # Training the model
            lstm.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'],)
            lstm.fit(tweet_train,label_train,epochs=10,batch_size=32)
            clf = lstm
            with open('LSTM.pickle','wb') as f:
                pickle.dump(clf,f)        
    for test in testsets.testsets:
        id_test,sentiment_test,tweet_test = dataset(test)
        if classifier == 'LSTM':   
            max_words = 5000
            dim = 100
            tokenizer = Tokenizer(num_words = 2000)
            tokenizer.fit_on_texts(tweet_test)
            sequences = tokenizer.texts_to_sequences(tweet_test)
            tweet_test = pad_sequences(sequences, maxlen=dim)
            label_test = []
            for i in sentiment_test:
                if i == 'negative':
                    i = 0
                elif i == 'neutral':
                    i = 1
                else :
                    i = 2
                label_test.append(i)
            label_test = keras.utils.to_categorical(label_test, num_classes=3)
            numclasses = clf.predict_classes(tweet_test)
            predicted = []
            for i in numclasses:
                if i == 0:
                    i = 'negative'
                elif i == 1:
                    i = 'neutral'
                else :
                    i = 'positive'
                predicted.append(i)
        else:
            predicted = clf.predict(tweet_test)    
        predictions={}
        for i in range(len(tweet_test)):
            predictions[id_test[i]]=predicted[i]
        evaluation.evaluate(predictions, test, classifier)
        evaluation.confusion(predictions, test, classifier)

