import pandas as pd
import numpy as np
from TextAugmentor import TextAugmentor as ta
import time

data = pd.read_csv('./data/data.csv')

data_aug = pd.DataFrame(columns=['Query','Action'])
querys={}

def augmenting(sen,ele):
    augmentor = ta(sen)
    augmentor.GenerateSentences()
    
    for item in augmentor.generated_sentences:
        querys[item]=ele



count=0
for sen in zip(data['Query'],data['Action']):
    #try:
        count += 1
        if count%25 == 0:
            time.sleep(120)
        augmenting(sen[0],sen[1])
    #except:
     #   print(count)

for key, value in querys.items():
    data = data.append(pd.DataFrame([list((key,value))],columns=['Query','Action']))



data.to_csv('data/processed_data.csv')