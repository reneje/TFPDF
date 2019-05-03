import pandas as pd 
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
import collections
from pandas import DataFrame
import re
import math
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.cluster import KMeans

def removeStop(document):
    removestopword = StopWordRemoverFactory()
    removestopwords = removestopword.create_stop_word_remover()
    return removestopwords.remove(document)

def stemming(document):
    #stemming proses
    stemmer = StemmerFactory()
    stemer = stemmer.create_stemmer()
    return stemer.stem(document)

def countFreqTerm(document):
    vocabs=[]
    count = CountVectorizer(min_df=0., max_df=1.0)
    x = count.fit_transform(document)
    counts = x.sum(axis=0).A1
    vocab = list(count.get_feature_names())
    # for voc in vocab:
    #     voc = removeStop(voc)
    #     vocabs.append(voc)
    vocabs = list(filter(None,vocab))
    freq_distribution = dict(zip(vocabs, counts))
    return freq_distribution,vocabs

def makeInvertedIndex(document):
    count = CountVectorizer(min_df=0., max_df=1.0)
    x = count.fit_transform(document)
    df = pd.DataFrame(x.A, columns=count.get_feature_names())
    return df

def countFreqDocument(vocab,dataVocab):
    total_document_contain ={}
    k = []
    for v in vocab:
        docOfFreq = dataVocab[v]
        k = [x for x in docOfFreq if x!= 0]
        total_document_contain[v] = len(k)
    return total_document_contain

def allTermTotal(freqTerm):
    totalAllfreq = 0
    # stopwords = open('stopwords_id.txt').read()
    # stopwords = list(stopwords.split('\n'))
    for key,value in sorted(freqTerm.items(), key=lambda item: (item[1], item[0]),reverse=True):
        # if key not in stopwords:
        kuadratvalue = pow(value,2)
        totalAllfreq = totalAllfreq + kuadratvalue
    return (np.sqrt(totalAllfreq))

def panjangVector(freqTerm,total_freq):
    pjgVektor={}
    for key,value in sorted(freqTerm.items(), key=lambda item: (item[1], item[0]),reverse=True):
        pjgVektor[key] = value/total_freq
    return pjgVektor
        
def word_tokenizer(text):
            #tokenizes and stems the text
    tokens = word_tokenize(text)
    return tokens

def cluster_sentences(sentences, nb_of_clusters,stopwords):
    tfidf_vectorizer = TfidfVectorizer(tokenizer=word_tokenizer,
                                    stop_words=stopwords,
                                    max_df=0.9,
                                    min_df=0.1,
                                    lowercase=True)
    #builds a tf-idf matrix for the sentences
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    kmeans = KMeans(n_clusters=nb_of_clusters)
    kmeans.fit(tfidf_matrix)
    clusters = collections.defaultdict(list)
    for i, label in enumerate(kmeans.labels_):
            clusters[label].append(sentences[i])
    return dict(clusters)

def saveclusterdata(n_clusters,hasil_kluster,files,link_sentence,unit_kata_contains,bobot_sentence,pilihcluster):
    for i in range (0,n_clusters):
        sentences = hasil_kluster[i]
        df = pd.DataFrame()
        top_kal = []
        word_top = []
        bobot_kata_in_sentence = []
        for k in sentences:
            fileku = files[(files['link'].str.contains(link_sentence[k]))]
            df = df.append(fileku)
            top_kal.append(k)
            unit_kata_contains[k]
            word_top.append(unit_kata_contains[k])
            bobot_kata_in_sentence.append(bobot_sentence[k])
        df['top kalimat'] = sentences
        df['word'] = word_top
        df['total bobot'] = bobot_kata_in_sentence
        df.to_csv(pilihcluster+"_"+str(i)+".csv",index=False)

#from stackoverflow
def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
    angle_product = dotproduct(v1, v2) / (length(v1) * length(v2))
    if angle_product > 1:
        angle_product = 1
    return math.acos(angle_product)
