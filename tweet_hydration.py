from datetime import datetime, timedelta
import tweepy
import json
import math
import glob
import csv
import zipfile
import zlib
import argparse
import os
import os.path as osp
import pandas as pd
from tweepy import TweepError
from time import sleep
from datetime import datetime, timedelta
import gzip
import shutil

# adapted from SMMT's get_metadata.py to be used in an IPython notebook with live command-line output

def get_metadata(input_file, output_file, id_column=None):
    if input_file is None or output_file is None:
        print("input_file or output_file arg cannot be None")
        return

    with open('api_keys.json') as f:
        keys = json.load(f)

    auth = tweepy.OAuthHandler(keys['consumer_key'], keys['consumer_secret'])
    auth.set_access_token(keys['access_token'], keys['access_token_secret'])
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    
    output_file_noformat = output_file.split(".",maxsplit=1)[0]
    print(output_file)
    output_file = '{}'.format(output_file) 
    ids = []
    
    if '.tsv' in input_file:
        inputfile_data = pd.read_csv(input_file, sep='\t')
        print('tab separated file, using \\t delimiter')
    elif '.csv' in input_file:
        inputfile_data = pd.read_csv(input_file)

    if not isinstance(id_column, type(None)):
        inputfile_data = inputfile_data.set_index(id_column)
    else:
        inputfile_data = inputfile_data.set_index('tweet_id')
    ids = list(inputfile_data.index)

    print('total ids: {}'.format(len(ids)))

    start = 0
    end = 100
    limit = len(ids)
    i = int(math.ceil(float(limit) / 100))

    last_tweet = None
    if osp.isfile(output_file):
        with open(output_file, 'rb') as f:
            #may be a large file, seeking without iterating
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
            last_line = f.readline().decode()
        last_tweet = json.loads(last_line)
        start = ids.index(last_tweet['id'])
        end = start+100
        i = int(math.ceil(float(limit-start) / 100))

    try:
        with open(output_file, 'a') as outfile:
            req_start = datetime.now()
            for go in range(i):
                if go % 500 == 0:
                    print('({}) currently on tweet {}/{}'.format(datetime.now().strftime('%x %X'), start + 1, limit))
                id_batch = ids[start:end]
                start += 100
                end += 100
                tweets = api.statuses_lookup(id_batch, tweet_mode='extended')
                for tweet in tweets:
                    json.dump(tweet._json, outfile)
                    outfile.write('\n')
                    
            return True
    except Exception as e:
        print('an exception occurred in get_metadata, exiting prematurely')
        print(e)
        return False

def get_date_to_hydrate():
    with open('next_hydrate.txt', 'r') as next_hydrate:
        return next_hydrate.read()
    
def timestamp():
    return datetime.now().strftime('%x %X')

# hydrate entries from the date stored in date_str
paused = False
while get_date_to_hydrate() in os.listdir('covid19_twitter/dailies'):
    if len(os.listdir('hydrated')) < 5:
        paused = False
        date = get_date_to_hydrate()
        print('({}) Hydrating tweets for {}'.format(timestamp(), date))

        # unzip cleaned dataset
        unzipped_path = f'unzipped/{date}.tsv'
        with gzip.open(f'covid19_twitter/dailies/{date}/{date}_clean-dataset.tsv.gz', 'rb') as f_in:
            with open(unzipped_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # hydrate; inuse status tells to the database processing pipeline to wait to use a file that's currently being written to
        hydrated_path = f'hydrated/{date}-inuse.json'
        is_success = get_metadata(unzipped_path, hydrated_path)
        while not is_success:
            # retry if get_metadata fails
            is_success = get_metadata(unzipped_path, hydrated_path)

        finished_hydrated_path = f'hydrated/{date}.json'
        os.rename(hydrated_path, finished_hydrated_path)
        print('done creating {}'.format(finished_hydrated_path))

        # store the next day into next_hydrate.txt
        date = (datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        with open('next_hydrate.txt', 'w') as next_hydrate:
            next_hydrate.write(date)
    else:
        if not paused:
            print('({}) 5 hydrated datasets are ready to be processed into the database. Hydration paused.'.format(timestamp()))
            paused = True
        sleep(1)
