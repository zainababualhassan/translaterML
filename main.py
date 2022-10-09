# load Flask
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
import regex as re


def remove_punctuation(text):
    return re.sub(r'[.!?:;,]', '', text)


english = pd.read_csv('en.csv', header=None)
french = pd.read_csv('fr.csv', header=None)
df = pd.concat([english, french], axis=1)
df.columns = ['English', 'French']
df['English'] = df['English'].apply(lambda x: remove_punctuation(x))
df['French'] = df['French'].apply(lambda x: remove_punctuation(x))
eng_tokenizer = Tokenizer()
eng_tokenizer.fit_on_texts(df['English'])
fr_tokenizer = Tokenizer()
fr_tokenizer.fit_on_texts(df['French'])


def final_predictions_model(sentence):
    y_id_to_word = {value: key for key, value in fr_tokenizer.word_index.items()}
    y_id_to_word[0] = ''

    sentence = eng_tokenizer.texts_to_sequences([sentence])

    sentence = pad_sequences(sentence, maxlen=15, padding='post')
    predictions = model.predict(sentence)
    return ' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]])


english_tokenizer = Tokenizer(num_words=200)
french_tokenizer = Tokenizer(num_words=323)

app = Flask(__name__)
model = load_model('model/model.h5')


# define a predict function as an endpoint
@app.route("/", methods=["GET", "POST"])
def index_view():
    return render_template('index.html')

@app.route('/translate',methods=['GET','POST'])
def translate():
    data = {"success": False}
    # get the request parameters
    params = request.json
    if params is None:
        params = request.args
    # if parameters are found, echo the msg parameter
    if params is not None:
        data["english"] = params.get("eng")
        print(data["english"])
        data["success"] = True
        data['french'] = final_predictions_model(data["english"])
        print(data["french"])
    # return a response in json format
    return render_template('index.html', eng=data['english'], fr=data['french'].encode('utf-8'),
                           enctype="multipart/form-data")


# start the flask app, allow remote connections
app.run(host='0.0.0.0',port=5000)
