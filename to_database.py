import spacy
import twarc
import tweepy
import argparse
from xtract import xtract
import numpy as np
import pandas as pd
import os
import gzip
import re
import json
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sqlalchemy import create_engine
from time import sleep
from datetime import datetime

# We first define some criteria for Tweets that we include.
# (i.e. that we have strong enough evidence to say that the Tweet is in English and from the USA)
NUM_CITIES = 1000
us_cities_df = pd.read_csv('us_cities.csv')['city'][:NUM_CITIES]

NUM_WORLD_CITIES = 600
world_cities = pd.read_csv('worldcities.csv')[:NUM_WORLD_CITIES]
world_cities = world_cities[world_cities['iso2'] != 'US']['city_ascii']

canada_territories = ['alberta', 'british', 'columbia', 'manitoba', 'new brunswick', 'newfoundland', 'labrador', 'northwest territories', 'nova scotia', 'nunavut', 'ontario', 'prince edward island', 'quebec', 'saskatchewan', 'yukon']
australia_territories = ['new south wales', 'queensland', 'south australia', 'tasmania', 'victoria', 'western australia']

us_cities = set(us_cities_df.to_list()) - set(world_cities.to_list()) - set(canada_territories) - set(australia_territories)
# print('Number of cities in list:', len(us_cities))
# print()

excluded = set(us_cities_df.to_list()).intersection(set(world_cities.to_list()))
# print('Cities excluded for being ambigiously in the US:')
# print(excluded)
# print()

# Birmingham and San Jose are fairly large US cities so add them back
us_cities = [x.lower() for x in us_cities.union({'Birmingham', 'San Jose'}) - set('Bristol')]

states_df = pd.read_csv('state_abbrevs.csv')
states = [x.lower() for x in states_df['state'].to_list()]
abbrev_regexes = [re.compile(r'(^{}$)|(\W{}$)'.format(x.lower(), x.lower())) for x in states_df['abbreviation'].to_list()]

# print(states)
# print()
# print(abbrev_regexes)

usa = ['united states', 'america']
USA = ['usa', 'united states', 'america']

def is_en_us(data):
    if data is None:
        return False
    
    if (data['lang'] != 'en'):
        return False
    
    if (data['place'] is not None and data['place']['country_code'] == 'US'):
        return True
    
    location = data['user']['location'].lower()
    if any(x in location for x in USA):
        return True
    if any(state in location for state in states):
        return True
    if any(us_city in location for us_city in us_cities):
        return True
    if any(abbrev_regex.search(location) for abbrev_regex in abbrev_regexes):
        return True
    
    description = data['user']['description'].lower()
    if any(x in description for x in usa):
        return True
    if any(state in description for state in states):
        return True
    if any(us_city in description for us_city in us_cities):
        return True
    
    return False

# Here, we run our hydrated datasets through this filter and into a database.
# We will compute VADER sentiment values along the way.

def timestamp():
    return datetime.now().strftime('%x %X')

analyzer = SentimentIntensityAnalyzer()
url_regex = re.compile(r'^https?:\/\/.*[\r\n]*')

engine = create_engine(r'sqlite:///C:\Users\epicp\Documents\PRINCETON_F20\independent-work\src\database\tweets.db')
conn = engine.connect()

file_format_re = re.compile(r'^[0-9]{4}-[0-9]{2}-[0-9]{2}\.json$')
paused = False
while True:
    hydrated_files = list(filter(lambda f: file_format_re.match(f), os.listdir('hydrated')))
    if len(hydrated_files) != 0:
        filename = hydrated_files[0]
        paused = False
        with open('./hydrated/{}'.format(filename), 'r') as f:
            print('({}) Started loading {} into database'.format(timestamp(), filename))
            print()

            print('({}) Reading json input file... '.format(timestamp()), end='', flush=True)
            data = filter(is_en_us, (json.loads(line) for line in f))
            print('Done!')

            print('({}) Creating DataFrame... '.format(timestamp()), end='', flush=True)
            df_all = pd.json_normalize(data, max_level=1)
            df = df_all[['created_at', 'id_str', 'full_text', 'user.id_str', 'user.followers_count', 'user.screen_name', 'user.verified', 'retweet_count', 'favorite_count']]
            df['created_at'] = df['created_at'].apply(pd.to_datetime)
            print('Done!')

            # remove URLs from full_text, get VADER scores, flatten into one column for each score
            print('({}) Generating neg/neu/pos/compound sentiment intensity scores... '.format(timestamp()), end='', flush=True)
            df = pd.concat([df, df['full_text'].apply(lambda x: url_regex.sub('', x)).apply(analyzer.polarity_scores).apply(pd.Series)], axis=1)
            print('Done!')

            # load extracted data in DataFrame into database
            df.to_sql('tweets', conn, if_exists='append')

        # delete hydrated data once we're done using it because the files are huge
        os.remove('./hydrated/{}'.format(filename))
    else:
        if not paused:
            print('({}) No more hydrated data to load into database. Script paused.'.format(datetime.now().strftime('%x %X')))
            paused = True
        sleep(1)

conn.close()
