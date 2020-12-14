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

import pyLDAvis

NUM_DAYS = 241

#####################################
##### PREPROCESSING DTM DATASET #####
#####################################

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

print('({}) DTM training data preprocessing started'.format(timestamp()))
start = datetime.now()
orig_df = pd.read_pickle(r'dfs\2020-03-22-to-2020-11-18-1000-daily')
orig_texts = [preprocess(text) for text in orig_df['full_text']]
orig_dictionary = corpora.Dictionary(orig_texts)
orig_corpus = [orig_dictionary.doc2bow(text) for text in orig_texts]

dtm_model = DtmModel.load(r'dtm\2020-03-22-to-2020-11-18-1000-daily')
print('Time to preprocess training texts:', str(datetime.now() - start))

######################################
##### DAY-BY-DAY TOPIC LABELLING #####
######################################

conn = sqlite3.connect('database/tweets.db')

# df = pd.read_pickle(r'dfs\2020-03-22-to-2020-08-19-4000-daily')
# df = pd.read_sql_query('select * from tweets where "user.screen_name" in (select screen_name from labels) and created_at between \'2020-03-22\' and \'2020-11-18\'', conn)

# comment out n_tweets_per_day lines and cumulative_tweets lines when using full dataset
# n_tweets_per_day = df['created_at'].apply(lambda x: x[:10]).value_counts().sort_index().values.tolist()

# cumulative_tweets[i] is the index of the first Tweet from i days after START_DATE
# cumulative_tweets = [sum(n_tweets_per_day[:i]) for i in range(NUM_DAYS + 1)]

for i in range(NUM_DAYS):
    ### GET AND PREPROCESS [FULL OR PARTIAL] DATASET
    start_date = (datetime(2020, 3, 22) + timedelta(days=i)).strftime('%Y-%m-%d')
    end_date = (datetime(2020, 3, 22) + timedelta(days=i + 1)).strftime('%Y-%m-%d')
    print('\n({}) Started topic labelling on Tweets from {}'.format(timestamp(), start_date))
    df = pd.read_sql_query("select * from tweets where id_str in (select id_str from tweets where created_at between '{}' and '{}' order by random() limit 25000)".format(start_date, end_date), conn)
    # texts = [preprocess(text) for text in df['full_text'][cumulative_tweets[i]:cumulative_tweets[i+1]]]
    texts = [preprocess(text) for text in df['full_text']] # full dataset version
    print('({}) Done with preprocessing!'.format(timestamp()))

    ### CONSTRUCT LIST OF DOCUMENT TOPICS
    max_likelihood_topics = []

    doc_topic, topic_term, doc_lengths, term_frequency, vocab = dtm_model.dtm_vis(time=i, corpus=orig_corpus)
    
    num_words = term_frequency.sum()
    
    # this is how pyLDAvis calculates topic proportions
    topic_freq = doc_topic.T @ doc_lengths
    P_t = topic_freq / topic_freq.sum()
    
    # this is what makes mathematical sense to me but it's much slower and close enough to pyLDAvis's method
    # orig_corpus_nonempty = list(filter(lambda x: len(x) != 0, orig_corpus))
    # P_t = np.sum([doc_topic[i] * np.prod([(term_frequency[w] / num_words) ** f for w, f in orig_corpus_nonempty[i]]) for i in range(len(orig_corpus_nonempty))], axis=0)
    # P_t = P_t / P_t.sum()
    
    word_freq = P_t.T @ topic_term
    for text in texts:
        D = orig_dictionary.doc2bow(text)
        log_doc_topic_dist = np.log(P_t) + np.sum([f * np.log(topic_term[:, w]) for w, f in D], axis=0)
        max_likelihood_topics.append(int(np.argmax(log_doc_topic_dist) + 1))

    print('({}) Done generating list of topics!'.format(timestamp()))

    ### WRITE TO DATABASE
    # ids = list(df['id_str'][cumulative_tweets[i]:cumulative_tweets[i+1]])
    ids = list(df['id_str']) # full dataset version
    assert(len(ids) == len(max_likelihood_topics))
    assert(isinstance(max_likelihood_topics[0], int))
    topics_df = pd.DataFrame({'id_str': ids, 'topics': max_likelihood_topics})
    print(topics_df.sample(3)) # for testing
    topics_df.to_sql('topics', conn, if_exists='append')
    print('({}) Done writing to database!'.format(timestamp()))
