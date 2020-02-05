
from numpy.random import seed
seed(1)
base_folder="C:\\Indranil\\HylandLab\\Code\\gitPublic\\Text_Summerizer\\"
modelLocation=base_folder+"TrainedModels\\"

import gensim as gs
import pandas as pd
import numpy as np
import scipy as sc
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.sequence import pad_sequences
import logging
import re
from collections import Counter

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)
    
emb_size_all = 300
maxcorp=5000


def createCorpus(article_and_summary_data_dic):
    corpus = []
    all_sentences = []
    # this two for loop will extract all the sentences
    for dictionary_key in article_and_summary_data_dic:
        for item in article_and_summary_data_dic[dictionary_key]:
            tokenized_item=sent_tokenize(item) #sent_tokenize is sentence tokenizer
            corpus.append(tokenized_item)
    
    # At this point corpus is a 2D array . 
    # No. of Row = 2*no.of article/summary files this should b the value of corpus_len
    # Each row is a vector which contains sentences in that file. In other word it is another array            
    corpus_len=len(corpus)
    for file_item in range(corpus_len):
        for sentence in corpus[file_item]:
            # We are converting 2D sentence array to 1D sentence array
            all_sentences.append(sentence)
    
    all_setence_len = len(all_sentences)       
    for m in range(all_setence_len):
        all_sentences[m] = word_tokenize(all_sentences[m])
    
    all_words=[]
    for sent in all_sentences:
        hold=[]
        for word in sent:
            hold.append(word.lower())
        all_words.append(hold)
        
    # all_words 2D array
    # No. of row=no. of sentences
    # i th row will have column, of length equals to no. of words in i th sentence.
    return all_words


def word2vecmodel(corpus):
    emb_size = emb_size_all
    model_type={"skip_gram":1,"CBOW":0}
    window=10
    workers=4
    min_count=4
    batch_words=20
    epochs=25
    #include bigrams
    #bigramer = gs.models.Phrases(corpus)

    model=gs.models.Word2Vec(corpus,size=emb_size,sg=model_type["skip_gram"],
                             compute_loss=True,window=window,min_count=min_count,workers=workers,
                             batch_words=batch_words)
        
    model.train(corpus,total_examples=len(corpus),epochs=epochs)
    model.save("%sWord2vec"%modelLocation)
    print('\007')
    return model


def summonehot(all_summaries):
    
    allwords=[]
    annotated={}
    for sent in all_summaries:
        for word in word_tokenize(sent):
            allwords.append(word.lower())
    
    #allwords is 1D array contains all words,which are present in all summary files
    print(len(set(allwords)), "unique characters in all_summaries")
    #maxcorp=int(input("Enter desired number of vocabulary: "))
    #maxcorp represnts no. of words we want to output.
   # maxcorp=int(len(set(allwords))/1.1)
    maxcorp=10
    wordcount = Counter(allwords).most_common(maxcorp)
    allwords=[]
    
    for p in wordcount:
        allwords.append(p[0])  
        
    allwords=list(set(allwords))
    
    print(len(allwords), "unique characters in all_summaries after max all_summaries cut")
    #integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(allwords)
    #one hot
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    #make look up dict
    for k in range(len(onehot_encoded)): 
        inverted = cleantext(label_encoder.inverse_transform([argmax(onehot_encoded[k, :])])[0]).strip()
        annotated[inverted]=onehot_encoded[k]
    return label_encoder,onehot_encoded,annotated



def wordvecmatrix(model,data):
    IO_data={"article":[],"summaries":[]}
    i=1
    for k in range(len(data["articles"])):
        art=[]
        summ=[]
        for word in word_tokenize(data["articles"][k].lower()):
            try:
                artval=model.wv.word_vec(word)
                art.append(artval)
            except Exception as e:
                print(e)

        for word in word_tokenize(data["summaries"][k].lower()):
            try:
                summ.append(onehot[word])
                #summ.append(model.wv.word_vec(word))
            except Exception as e:
                print(e)
        
        IO_data["article"].append(art) 
        IO_data["summaries"].append(summ)
        if i%100==0:
            print("progress: " + str(((i*100)/len(data["articles"]))))
        i+=1
    #announcedone()
    print('\007')
    return IO_data

def cutoffSequences(data,artLen,sumlen):
    data2={"article":[],"summaries":[]}
    for k in range(len(data["article"])):
        if len(data["article"][k])<artLen or len(data["summaries"][k])<sumlen:
             #data["article"]=np.delete(data["article"],k,0)
             #data["article"]=np.delete(data["summaries"],k,0)
             pass
        else:
            data2["article"].append(data["article"][k][:artLen])
            data2["summaries"].append(data["summaries"][k][:sumlen])
    return data2


def max_len(data):
    lenk=[]
    for k in data:
        lenk.append(len(k))
    print("The minimum length is: ",min(lenk))
    print("The average length is: ",np.average(lenk))
    print("The max length is: ",max(lenk))
    return min(lenk),max(lenk)

"""reshape vectres for Gensim"""
#def reshape(vec):
#    return np.reshape(vec,(1,emb_size_all))

def addones(seq):
    return np.insert(seq, [0], [[0],], axis = 0)

def endseq(seq):
    pp=len(seq)
    return np.insert(seq, [pp], [[1],], axis = 0)
#######################################################################
    
corpus = createCorpus(data)

label_encoder,onehot_encoded,onehot=summonehot(data["summaries"])

model=word2vecmodel(corpus)

model.get_latest_training_loss()

train_data = wordvecmatrix(model,data)

print(len(train_data["article"]), "training articles")

train_data=cutoffSequences(train_data,300,10)

#seq length stats
max_len(train_data["article"])
max_len(train_data["summaries"])


train_data["summaries"]=np.array(train_data["summaries"])
train_data["article"]=np.array(train_data["article"])


#add end sequence for each article

#train_data["summaries"]=np.array(list(map(endseq,train_data["summaries"])))
#train_data["article"]=np.array(list(map(endseq,train_data["article"])))

print("summary length: ",len(train_data["summaries"][0]))
print("article length: ",len(train_data["article"][0]))


"""__pad sequences__
train_data["article"]=pad_sequences(train_data["article"],maxlen=max_len(train_data["article"]),
          padding='post',dtype=float)
train_data["summaries"]=pad_sequences(train_data["summaries"],maxlen=max_len(train_data["summaries"]),
          padding='post',dtype=float)
"""
#add start sequence
train_data["summaries"]=np.array(list(map(addones,train_data["summaries"])))