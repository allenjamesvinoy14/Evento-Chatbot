import Evento as ev
from flask import Flask,request,render_template
import keras.models
from load import *
from keras.preprocessing.sequence import pad_sequences
import numpy as np



app = Flask(__name__)

global model,graph
model,graph = init()

evento_obj = ev.Evento()
evento_obj.process_data()

def get_prediction(query):
    
    mat = evento_obj.get_embed_matrix(query)
    x = pad_sequences(maxlen=18, sequences=[mat], padding="post", value=evento_obj.vocab.index('PAD'))
    ans = np.argmax(model.predict(x)[0])
    return evento_obj.action_index_1[ans]



@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
def predict():

    query = request.form['query']
    with graph.as_default():

        action = get_prediction(query)
        return render_template("predict.html",output=action)




if __name__ == '__main__':
    app.run(port = 33507, debug = True)