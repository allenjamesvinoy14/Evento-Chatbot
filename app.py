from flask import Flask,abort
import pandas as pd
import numpy as np
import nltk
from keras.models import model_from_json

data = pd.read_csv('./data/processed_data.csv')

vocab = []

for index,row in data.iterrows():
    
    tokens = nltk.word_tokenize(row['Query'])
    for i in tokens :
        
        if i not in vocab:
            
            vocab.append(i)
vocab.append('UNK') #Unknown token
vocab.append('PAD') #Pad token
n_words = len(vocab)
actions = list(data['Action'].unique())
n_actions = len(actions)

action_index_1 = {}
action_index_2 = {}

for i,v in enumerate(actions):
    action_index_1[i] = v
    action_index_2[v] = i

def get_embed_matrix(sentence):
    
    embeds = []
    tokens = nltk.word_tokenize(sentence)
    
    for i in tokens:
        
        if i in vocab:
            
            n = vocab.index(i)
        else :
            
            n = vocab.index('UNK')
        embeds.append(n)
    return np.array(embeds)



json_file = open('Model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("Model/model.h5")
print("Loaded model from disk")

app = Flask(__name__)

@app.route('/api', methods=['POST'])

def get_prediction(query):
    
    mat = get_embed_matrix(query)
    x = pad_sequences(maxlen=18, sequences=[mat], padding="post", value=vocab.index('PAD'))
    ans = np.argmax(model.predict(x)[0])
    return action_index_1[ans]

if __name__ == '__main__':
    app.run(port = 9000, debug = True)