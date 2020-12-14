import sqlite3
import pandas as pd
import numpy as np
import re
import unicodedata
from datetime import datetime, timedelta

from string import punctuation

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from gensim import corpora
# from gensim.models import LdaMulticore
from gensim.models.wrappers.dtmmodel import DtmModel
from gensim.test.utils import common_corpus
from gensim.models.coherencemodel import CoherenceModel

from tmtoolkit.topicmod.evaluate import metric_coherence_gensim

import pickle

################################################################################

sample_path = r'dfs\2020-03-22-to-2020-11-18-1000-daily'
dtm_out_path = 'dtm' + sample_path[3:]

conn = sqlite3.connect('database/tweets.db')

# comment out if new sample is needed
df = pd.read_sql_query("select * from tweets where id_str in (select id_str from tweets where created_at between '2020-03-22' and '2020-11-18' order by random() limit 242000)", conn)
df.to_pickle(sample_path)

df = pd.read_pickle(sample_path)

################################################################################

# regex from https://stackoverflow.com/questions/3809401/what-is-a-good-regular-expression-to-match-a-url
url_re = re.compile(r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})")
mentions_re = re.compile(r"@\w*")
hashtag_re = re.compile(r"#\w*")
stopword_list = stopwords.words('english')
wnl = WordNetLemmatizer()

# From https://www.kaggle.com/alvations/basic-nlp-with-nltk#Stemming-and-Lemmatization
def penn2morphy(penntag):
    # Converts Penn Treebank tags to WordNet.
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n' # if mapping isn't found, fall back to Noun.

def preprocess(text):
    text = url_re.sub('', text)
    text = mentions_re.sub('', text)
    text = hashtag_re.sub('', text)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    # Tokenization and word removal
    words = word_tokenize(text)
    words = map(lambda w: w.lower(), words)
    words = filter(lambda w: w not in stopword_list and w not in punctuation and len(w) >= 3, words)
    words = list(words)

    # POS tagging and lemmatization
    tagged_words = pos_tag(words)
    lemmatized = [wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) for word, tag in tagged_words]
    lemmatized = list(filter(lambda w: not any(p in w for p in punctuation) and w not in stopword_list and w not in punctuation and len(w) >= 3, lemmatized))
    return lemmatized

def timestamp():
    return datetime.now().strftime('%x %X')

print('({}) Preprocessing/getting preprocessed data.'.format(timestamp()))

# preprocessed corpus already exists
# with open(r'dtm\full-preprocessed-pickle', 'rb') as f:
#     texts = pickle.load(f)

# actually does preprocessing
start = datetime.now()
texts = [preprocess(text) for text in df['full_text']]
print('Time to preprocess Tweets:', str(datetime.now() - start))

# save preprocessed corpus
with open(r'dtm\full-preprocessed-pickle', 'wb') as f:
    pickle.dump(texts, f)

# get time slices (number of tweets each day)
time_slices = df['created_at'].apply(lambda x: x[:10]).value_counts().sort_index().values.tolist()

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

################################################################################

dtm_exe_path = r'C:\Program Files\DTM\dtm-win64.exe'

print('({}) Model started training'.format(timestamp()))
start = datetime.now()
dtm_model = DtmModel(dtm_exe_path, corpus=corpus[:], time_slices=time_slices, num_topics=20, id2word=dictionary)
elapsed = datetime.now() - start
print('({}) Model finished training'.format(timestamp()))
print('Elapsed time:', elapsed)

print('Saving model...')
dtm_model.save(dtm_out_path)
print('({}) Model saved'.format(timestamp()))
