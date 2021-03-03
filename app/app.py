from flask import Flask, request, render_template, jsonify, url_for, send_file, flash, redirect
import pickle, os, datetime, json, re, joblib, cyrtranslit, requests, linecache, emoji
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'data'
app.config['MAX_CONTENT_PATH'] = 2000
app.secret_key = "1234567"

WORD_MODEL = 'data/model/word_model.vec'
WIKI_MORPH = 'data/wikimorph/wikimorph'

# Generate word to line index mapping from a word-vector file
def wv2fi(fpath):
    mapping = dict()
    with open(fpath, 'r', encoding='utf-8') as f:
        info = f.readline()
        length = int(info.split()[0])
        for i in range(1, length):
            l = f.readline().split()
            if len(l) > 0:
                mapping[l[0]] = i+1
    return mapping


mapping = wv2fi(WORD_MODEL)
morph = wv2fi(WIKI_MORPH)

def lemmatize(word, tag):
    if word.isnumeric():
        res = "{}\t{}\t{}".format(word, 'Num', word)
        return res
    res = "{}\t{}\t{}".format(word, tag, word)
    i = morph.get("{}_{}".format(word, tag))
    if i:
        line = linecache.getline(WIKI_MORPH, i).strip()
        lemma = line.split()[1]
        lemma = lemma.split(',')[0]
        if lemma:
            return lemma
    return word

# Clean text (applicable to tweets as well)
def clean(txt, tagger, le):
    tweet_text = re.sub("@[A-Za-z0-9_-]+", "", txt)  # remove mentions
    tweet_text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+",
                        "", tweet_text)  # remove links
    tweet_text = " ".join(tweet_text.split())
    tweet_text = ''.join(
        c for c in tweet_text if c not in emoji.UNICODE_EMOJI)  # remove emojis
    tweet_text = tweet_text.replace("#", "")  # remove hashtags
    tweet_text = re.sub(r'[^\w\s]', '', tweet_text)  # remove punctuation
    tweet_text = tweet_text.replace("_", " ")
    tweet_text = tweet_text.replace('RT ', '')  # remove retweets
    # convert to latin scrypt
    tweet_text = [cyrtranslit.to_latin(w.lower()) for w in tweet_text.split()]
    results = list()
    for word in tweet_text:
        i = mapping.get(word)
        if i:
            line = linecache.getline(WORD_MODEL, i).strip()
            vec = np.array([float(n) for n in line.split()[1:]])
            pred = le.inverse_transform(tagger.predict([vec]))[0]
            result = lemmatize(word, pred)
            results.append(result)
    if not results:
        return ''
    return ' '.join(results)

# Convert text to the list of vectors
def transform(txt, tagger, le):
    text = clean(txt, tagger, le)
    vecs = list()
    if text == '':
        return None
    text = text.split()
    for t in text:
        i = mapping.get(t)
        if i:
            line = linecache.getline(WORD_MODEL, i).strip()
            vec = np.array([float(n) for n in line.split()[1:]])
            vecs.append(vec)
    return vecs

# KNC for prediction
def loadClassifier():
    knn = joblib.load('data/model/knc.joblib')
    return knn

# MLP for tagging
def loadTagger():
    tagger = joblib.load('data/model/mlp.joblib')
    le = joblib.load('data/model/labels.joblib')
    return tagger, le


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    knn = loadClassifier()
    tagger, le = loadTagger()
    input_text = request.form['text_field']
    if len(input_text) > 30:
        flash("Unesite tekst do maksimalno 30 karaktera ili koristite učitavanje fajla.")
        return redirect(url_for('home'))
    transformed = transform(input_text, tagger, le)
    if transformed:
        vec = np.mean(transformed, axis=0).reshape(1, -1)
        pred = knn.predict(vec)[0]
        if bool(pred):
            response = "'{}' --> GOVOR MRŽNJE".format(input_text)
        else:
            response = "'{}' --> OK".format(input_text)
        flash(response)
    else:
        flash("Zadate reči nisu pronađene u modelu.")

    return redirect(url_for('home'))

# Upload file, process and return it for download
@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        if f.filename.split('.')[1] != 'txt':
            flash("Molimo učitajte isključivo .txt fajlove.")
            return redirect(url_for('home'))
        path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(path)

        with open(path, 'r', encoding='utf-8') as f:
            uploaded = f.read()
       
        knn = loadClassifier()
        tagger, le = loadTagger()
        
        result = "Prediction\tText\n"
        lines = uploaded.split('\n')
        if len(lines) > 30:
            flash("Molimo učitajte fajl do maksimalno 30 paragrafa.")
            return redirect(url_for('home'))
        for line in lines:
            transformed = transform(line, tagger, le)
            if not transformed:
                continue
            vec = np.mean(transformed, axis=0).reshape(1, -1)
            pred = knn.predict(vec)[0]
            result += str(bool(pred))+'\t'+line+'\n'
        os.remove(path)
        path = path.split('.txt')[0]+'.csv'
        with open(path, 'w+', encoding='utf-8') as f:
            f.write(result)
        
        return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run()
