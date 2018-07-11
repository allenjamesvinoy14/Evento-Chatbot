from flask import Flask,abort
import pandas as pd
import numpy as np
import nltk
from keras.models import model_from_json
import functions
from keras.preprocessing.sequence import pad_sequences



json_file = open('Model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("Model/model.h5")
print("Loaded model from disk")

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    return 'Hello, World!'



@app.route('/api', methods=['GET', 'POST'])

def get_prediction(query):
    query = "where is the food from?"
    mat = functions.get_embed_matrix(query)
    x = pad_sequences(maxlen=18, sequences=[mat], padding="post", value=vocab.index('PAD'))
    ans = np.argmax(model.predict(x)[0])
    return action_index_1[ans]





if __name__ == '__main__':
    app.run( debug = True)