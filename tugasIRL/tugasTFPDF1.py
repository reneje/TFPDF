import pandas as pd 
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import operator
import math
import numpy as np
import pickle
from sklearn.cluster import KMeans
import prosesTFPDF as pt
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import collections


files = pd.read_csv('data/cobadataspesifik.csv')
stopwords = open("stopwords-id.txt").read().split("\n")
links = list(files['link'])
# print(files)
# removestopword = StopWordRemoverFactory()
# removestopwords = removestopword.create_stop_word_remover()

source = list(files['source'])
# source = list(filter(None,source))
# print (source)
jenis_source = list(set(source))
print(jenis_source)
# df = pd.read_csv('data/dataIRL.csv')
# title = list(df['title'])
# news = list(df['news'])

nilai_tf_pdf = []
allvocab = []
data = {}
#dijalankan persource
for i in jenis_source:
    datapersource = []
    data[i] = files [(files['source'].str.contains(i))]
    # judul = list(data[i]['title'])
    news = list(data[i]['news'])
    # print(len(judul))
    #mencari frekuensi kata
    for j in range(0,len(news)):

        # news[j] = re.sub('.*?CNN Indonesia --',' ',news[j],flags=re.I)
        # news[j] = re.sub('.*?KOMPAS.com -',' ',news[j],flags=re.I)
        # news[j] = re.sub('.*?KOMPAS. com',' ',news[j], flags=re.I)
        news[j] = re.sub('Kompas.com','Kompas_com',news[j], flags=re.I)
        news[j] = re.sub('\d+','',news[j])
        news[j] = re.sub(r'[.,\"^$*+?\{\}\-\[\]\\/|()]',' ',news[j])
        # judul[j] = re.sub(r'[.,\"^$*+?\{\}\-\[\]\\/|()]',' ',judul[j])
        # datapersource.append(judul[j].lower()+" "+news[j].lower())
        datapersource.append(news[j])
    #  print(datapersource)
    # datapersource = pt.cleaning(datapersource)
    df = pt.makeInvertedIndex(datapersource)
    frequencyOfTerm, vocab= pt.countFreqTerm(datapersource)
    # print(frequencyOfTerm)
    totalfreq = pt.allTermTotal(frequencyOfTerm)
    print(totalfreq)
    fkc = pt.panjangVector(frequencyOfTerm,totalfreq)
    # print(fkc)
    freqDocument = pt.countFreqDocument(vocab,df)
    totalDoc = len(news)

    # menghapus kata stopword dengan sastrawi
    # wordnotstp=[]
    # for word in vocab:
    #     wordnotstp.append(removestopwords.remove(word))
    # wordnotstp = list(filter(None,wordnotstp))
    
    #menghapus kata stopword
    
    wordnotstp=[]
    for word in vocab:
        if word not in stopwords:
            wordnotstp.append(word)

    #mengitung bobot kata pada satu source berita
    weight={}
    for i in wordnotstp:
        weight[i] = frequencyOfTerm[i] * np.exp2(freqDocument[i]/totalDoc)

    nilai_tf_pdf.append(weight)
# allvocab = list(set(allvocab))

### pada bagian ini menjumlahkan seluruh bobot kata pada source berita
sumweigth = {}
for i in nilai_tf_pdf:
    c = list(i.keys())
    for j in c:
        sumweigth[j] = 0

for i in nilai_tf_pdf:
    c = list(i.keys())
    for j in c:
        sumweigth[j] += i[j]
count = 0
getscore = open("TopKata50.txt","w+")
bobotkata={}
for key,value in sorted(weight.items(), key=lambda item: (item[1], item[0]),reverse=True):
    if key is not None:
        
        getscore.writelines(str(value)+" "+str(key)+"\n")
        
        bobotkata[key] = value
        
        if count == 49:
            break
        count+=1
getscore.close()

# saving
# with open ('bobotkata_top30_1.pickle', 'wb') as f_buff : 
#     pickle.dump (bobotkata, f_buff) 

# loading
# bobotkata = pickle.load (open ('bobotkata_top30_1.pickle', 'rb'))



# title = list(files['title'])
news = list(files['news'])
# files = open('scorelagi.txt').read().split("\n")
# bobotkata={}
# for f in files:
#     kata = f.split("   ")
#     # print(kata)
#     bobotkata[kata[1]] = float(kata[0])


# print(bobotkata)

#mencari bobot kalimat tertinggi perdocument.
highest_sentence_in_document=[]
for i in range (0, len(news)):
   
    if news[i] is not float :
        # news[i] = re.sub('.*?CNN Indonesia --',' ',news[i], flags=re.I)
        # news[i] = re.sub('.*?KOMPAS.com -',' ',news[i], flags=re.I)
        # news[i] = re.sub('.*?KOMPAS. com',' ',news[i], flags=re.I)
        news[i] = re.sub('Kompas.com','Kompas_com',news[i], flags=re.I)
        news[i] = re.sub (r'(\?|\.|\|)', r'\1 ', news[i])
        news[i] = re.sub (r'\ +', ' ', news[i])
        # sentence = title[i]+". "+news[i]
        sentence = news[i]
        # print("*"*30)
        # print(sentence)
        # with open ('document.pickle', 'wb') as f_buff : 
        #     pickle.dump (sentence, f_buff)
        sentences = sent_tokenize(sentence)
        
        banyakkata_persentence={}

        #cara mencarinya adalah dengan mencari kalimat dengan kata terbanyak yang terdapat dari kata Top-k
        
        for sen in sentences:
            # unit_vector=[]
            sumbanyakkata=0
            # total_bobot_sentence = 0
            tok = sen.lower().split(" ")
            
            for k in tok:
                k = re.sub(r'[.,\"^$*+?\{\}\-\[\]\\/|()]',"",k)
                # for kataTop in bobotkata.keys():
                if k in bobotkata.keys():
                    sumbanyakkata += 1
                    
                    # unit_vector.append(1)
                # else:
                #     unit_vector.append(0)
            # bobot_persentence[sen] = sum(unit_vector)
            banyakkata_persentence[sen] = sumbanyakkata
            
        # print(bobot_persentence)
        result = max(banyakkata_persentence.items(), key=operator.itemgetter(1))
        # for key,value in sorted(bobot_persentence.items(), key=lambda item: (item[1], item[0]),reverse=True):
        #     if key is not None:
        #         print(value," ",key)
    highest_sentence_in_document.append(result)

# saving
# with open ('highest_sent_top30_1.pickle', 'wb') as f_buff : 
#     pickle.dump (highest_sentence_in_document, f_buff) 

# loading open file
# highest_sentence_in_document = pickle.load (open ('highest_sent_top30.pickle', 'rb'))

#membuat unit vector dari sentence per document
all_sentence=[(x) for x, y in highest_sentence_in_document]
unit_vector_sentence={}
unit_kata_contains={}
bobot_sentence ={}
# k = 0
link_sentence={}
for i in range (0,len(all_sentence)):
    sentence = all_sentence[i]

    # sentence = re.sub(r"\[\]"," ",sentence)
    unit_vector=[]
    word_contains=[]
    sumbobot = 0
    # link = []
    all_sentence[i] = re.sub(r'[.,\"^$*+?\{\}\-\\/|()]'," ",all_sentence[i])
    k = all_sentence[i].lower().split(" ")
    for kataTop in bobotkata.keys():
        if kataTop in k:
            unit_vector.append(1)
            word_contains.append(kataTop)
            sumbobot += bobotkata[kataTop]
        else:
            unit_vector.append(0)
    # print(unit_vector)
    # if sum(unit_vector) != 0:
    # link.append(links[k])
    # links[i] = re.sub(r'\[\]',"",links[i])
    link_sentence[sentence]= links[i]
    # print(type(links[i]))
    unit_vector_sentence[sentence] = unit_vector 
    unit_kata_contains[sentence] = word_contains
    bobot_sentence[sentence] = sumbobot
 
values = np. matrix(list(unit_vector_sentence.values()))
sentence = list(unit_vector_sentence.keys())
X = values

################CLUSTERING###############################
n_clusters = 3
clusters = pt.cluster_sentences(sentence, n_clusters,stopwords)

kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
kelompok = list(kmeans.labels_)
kelompok_klustering ={}
for i in range (0,n_clusters):
    anggota_index =[]
    for k in range(0,len(kelompok)):
        if kelompok[k] == i:
            anggota_index.append(sentence[k])
    kelompok_klustering[i]=anggota_index

#dari kluster dengan TF*IDF
pt.saveclusterdata(n_clusters,clusters,files,link_sentence,unit_kata_contains,bobot_sentence,"hasilClusterKMeansTFIDF")
#dari kluster dengan unit vector
pt.saveclusterdata(n_clusters,kelompok_klustering,files,link_sentence,unit_kata_contains,bobot_sentence,"hasilClusterKMeansunitVector")

#rencana mau di kelompokkan tapi gatau
terdekat={}
for i in range (values.shape[0]-1):
    index_=[]
    for j in range (0, len(values)):
        vekt_i = np.array (values[i, :])[0]
        vekt_j = np.array (values[j, :])[0]
        radian = pt.angle(vekt_i, vekt_j)
        degree = math.degrees (radian)
        # print ("{} and {} : {:.2F}".format (i, j, degree))
        if degree < 41.45 and i != j:
            print(i," ",j)
            index_.append(j)
    terdekat[i] = index_

# # #set anggota awal untuk mencari centroid:

# kelompok={}
hasil_sentences = {}
no_cluster = 0
for i in terdekat.keys():
    # anggota = []
    sentence_anggota = []
    if len(terdekat[i]) > 10:
        print(i," ",terdekat[i])
        # anggota.append(i)
        sentence_anggota.append(sentence[i])
        for k in terdekat[i]:
            # anggota.append(k)
            sentence_anggota.append(sentence[k])
        # kelompok[no_cluster] = anggota
        hasil_sentences[no_cluster] = sentence_anggota
        no_cluster +=1

pt.saveclusterdata(len(hasil_sentences.keys()),hasil_sentences,files,link_sentence,unit_kata_contains,bobot_sentence,"hasilClusterunitVector")

