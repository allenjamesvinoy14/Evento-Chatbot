import pandas as pd
import numpy as np
from TextAugmentor import TextAugmentor as ta
import time
import nltk

data = pd.read_csv('./data/data.csv')

data_aug = pd.DataFrame(columns=['Query','Action'])
querys={}

def augmenting(sen,ele):
    augmentor = ta(sen)
    augmentor.GenerateSentences()
    
    for item in augmentor.generated_sentences:
        querys[item]=ele


print("Augmenting....")
count=0
for sen in zip(data['Query'],data['Action']):
    #try:
        count += 1
        if count%25 == 0:
            time.sleep(120)
            print("Done : " + str(count))
        augmenting(sen[0],sen[1])
    #except:
     #   print(count)

for key, value in querys.items():
    data = data.append(pd.DataFrame([list((key,value))],columns=['Query','Action']))

data = data.reset_index()

def process(s):
    w = nltk.word_tokenize(s)
    w = [i for i in w if i.isalpha() or i.isnumerical()]
    w = [i.lower() for i in w]
    return ' '.join(w)

print("Formatting to lower..")

processed = pd.DataFrame(columns=['Query','Action'])
for index,row in data.iterrows():
    processed.loc[index,'Query']=  process(row['Query'])
    processed.loc[index,'Action']= row['Action']
    if index%10==0:
        print("Processed : " + str(index))

print("Done")

processed.to_csv('data/processed_data.csv')