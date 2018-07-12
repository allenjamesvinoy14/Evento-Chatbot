import pandas as pd
import numpy as np
import nltk
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from flask import Flask, render_template



class Evento:


    def __init__(self):
            self.vocab = []
            self.action_index_1 = {}
            self.action_index_2 = {}

    def process_data(self):
        data = pd.read_csv('./data/processed_data.csv')

        for index,row in data.iterrows():
            
            tokens = nltk.word_tokenize(row['Query'])
            for i in tokens :
                
                if i not in self.vocab:
                    
                    self.vocab.append(i)
        self.vocab.append('UNK') #Unknown token
        self.vocab.append('PAD') #Pad token
        n_words = len(self.vocab)
        actions = list(data['Action'].unique())
        n_actions = len(actions)

        

        for i,v in enumerate(actions):
            self.action_index_1[i] = v
            self.action_index_2[v] = i

    def get_embed_matrix(self, sentence):
        
        embeds = []
        tokens = nltk.word_tokenize(sentence)
        
        for i in tokens:
            
            if i in self.vocab:
                
                n = self.vocab.index(i)
            else :
                
                n = self.vocab.index('UNK')
            embeds.append(n)
        return np.array(embeds)

