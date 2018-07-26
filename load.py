import keras.models  
import tensorflow as tf   
from keras.models import model_from_json

tf.reset_default_graph()
def init():

    json_file = open('Model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("Model/model.h5")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    graph = tf.get_default_graph()
    return model,graph
