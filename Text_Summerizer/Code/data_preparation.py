
import winsound as ws
import numpy as np
import os
import pandas as pd
import re

#####################define data sources#################
# datasource https://cs.nyu.edu/~kcho/DMQA/
base_folder="C:\\Indranil\\HylandLab\\Code\\gitPublic\\Text_Summerizer\\"
CNN_data=base_folder+"Data\\cnn\\"

datasets={"cnn":CNN_data}
data_categories=["training","validation","test"]
data={"articles":[],"summaries":[]}


def parsetext(dire,category,filename):
    with open("%s\\%s"%(dire+category,filename),'r',encoding="Latin-1") as readin:
        print(filename+"file read successfully")
        text=readin.read()
    return text.lower()


def load_data(dire,category):
    """dataname refers to either training, test or validation"""
    for dirs,subdr, files in os.walk(dire+category):
        filenames=files
    return filenames

def cleantext(text):
    text=re.sub(r"what's","what is ",text)
    text=re.sub(r"it's","it is ",text)
    text=re.sub(r"\'ve"," have ",text)
    text=re.sub(r"i'm","i am ",text)
    text=re.sub(r"\'re"," are ",text)
    text=re.sub(r"n't"," not ",text)
    text=re.sub(r"\'d"," would ",text)
    text=re.sub(r"\'s","s",text)
    text=re.sub(r"\'ll"," will ",text)
    text=re.sub(r"can't"," cannot ",text)
    text=re.sub(r" e g "," eg ",text)
    text=re.sub(r"e-mail","email",text)
    text=re.sub(r"9\\/11"," 911 ",text)
    text=re.sub(r" u.s"," american ",text)
    text=re.sub(r" u.n"," united nations ",text)
    text=re.sub(r"\n"," ",text)
    text=re.sub(r":"," ",text)
    text=re.sub(r"-"," ",text)
    text=re.sub(r"\_"," ",text)
    text=re.sub(r"\d+"," ",text)
    text=re.sub(r"[$#@%&*!~?%{}().,\`\'\"]"," ",text)
    
    return text

def announcedone():
    duration=2000
    freq=440
    ws.Beep(freq,duration)

def printArticlesum(k):
    print("---------------------original sentence-----------------------")
    print("-------------------------------------------------------------")
    print(data["articles"][k])
    print("----------------------Summary sentence-----------------------")
    print("-------------------------------------------------------------")
    print(data["summaries"][k])
    return 0


filenames=load_data(datasets["cnn"],data_categories[0])

"""----------load the data, sentences and summaries-----------"""
#for k in range(len(filenames)):
for k in range(len(filenames[:10000])):
        if k%2==0:
            try:
                parsed_data=parsetext(datasets["cnn"],data_categories[0],"%s"%filenames[k])
                cleaned_text=cleantext(parsed_data)
                #print("cleaned text of article"+ cleaned_text)
                data["articles"].append(cleaned_text)
            except Exception as e:
                data["articles"].append("Could not read")
                print(e)
        else:
            try:
                parsed_summary_data=parsetext(datasets["cnn"],data_categories[0],"%s"%filenames[k])
                cleaned_summary_text=cleantext(parsed_summary_data)
                #data["summaries"].append(cleantext(parsetext(datasets["cnn"],data_categories[0],"%s"%filenames[k])))
                data["summaries"].append(cleaned_summary_text)
            except Exception as e:
                data["summaries"].append("Could not read")
                print(e)

del filenames