from flask import Flask,abort



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


