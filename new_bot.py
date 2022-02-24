import datetime
from flask_caching import Cache
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
import plotly.express as px
from dash import Input, Output, State, dcc, html
external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css']
import plotly.express as px
from textblob import Word
from dash import dcc as dcc
from datetime import datetime
from io import StringIO
import pandas as pd
import numpy as np
import tweepy as tw
import re
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
lemma = WordNetLemmatizer()
stopit = stopwords.words('english')
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
lemmatizer = nltk.stem.WordNetLemmatizer()
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
from autocorrect import Speller
nltk.download('words')
words_me = set(nltk.corpus.words.words())
from wordcloud import WordCloud
import io
import s3fs
import sys
import os
import boto3
from io import StringIO
from smart_open import smart_open

import warnings
warnings.filterwarnings("ignore")


s3 = boto3.client('s3')
REGION = 'us-east-1'
ACCESS_KEY_ID = 'AKIA2K6MEWS7LIZQ2LO4'
SECRET_ACCESS_KEY = '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'
BUCKET_NAME = 'twitterdf'
cleandatame = 'twiiterdf/cleandata.csv'
filenetflix = 'twiiterdf/netflixdf.csv'
filehulu = 'twiiterdf/huludf.csv'
filetubi = 'twiiterdf/tubidf.csv'
data_key_net = 'twiiterdf/netflixdf.csv'
data_key_hul = 'twiiterdf/huludf.csv'
data_key_tub = 'twiiterdf/tubidf.csv'
data_key_clean = 'twiiterdf/cleandata.csv'
filenetflixclean = 'twiiterdf/netflixdfclean.csv'
filehuluclean = 'twiiterdf/huludfclean.csv'
filetubiclean = 'twiiterdf/tubidfclean.csv'
data_key_netclean = 'twiiterdf/netflixdfclean.csv'
data_key_hulclean = 'twiiterdf/huludfclean.csv'
data_key_tubclean = 'twiiterdf/tubidfclean.csv'

my_api_key = "Hzbb0sXPSUX5yFXRozdmdeqSX"
my_api_secret = "j2kIHynH5tqE4RV6lMC602K4HQKbTlVbTiguFj7R4UgeAHVpVV"
# authenticate
auth = tw.OAuthHandler(my_api_key, my_api_secret)
api = tw.API(auth, wait_on_rate_limit=False)

# # NETFLIX

# In[7]:


usernetflix = "netflix"
netflix = api.user_timeline(screen_name=usernetflix,
                            count=200,
                            include_rts=True,
                            tweet_mode='extended')
netflixdf = pd.DataFrame()

# In[8]:


usernetflix = "netflix"
netflix = api.user_timeline(screen_name=usernetflix,
                            count=200,
                            include_rts=True,
                            tweet_mode='extended')
netflixdf = pd.DataFrame()




for tweet in netflix:
    hashtags = []
    for hashtag in tweet.entities["hashtags"]:
        hashtags.append(hashtag["text"])
        text = api.get_status(id=tweet.id, tweet_mode='extended').full_text

    netflixdf = netflixdf.append(pd.DataFrame({'user_name': tweet.user.name,
                                                 'user_location': tweet.user.location,
                                                 'user_description': tweet.user.description,
                                                 'user_verified': tweet.user.verified,
                                                 'date': tweet.created_at,
                                                 'text': text,
                                                 'hashtags': [hashtags if hashtags else None],
                                                 're_tweet': tweet.retweet,
                                                 'source': tweet.source}))
    netflixdf = netflixdf.reset_index(drop=True)

csv_buffernet = StringIO()
netflixdf.to_csv(csv_buffernet, index=False)

s3csvnetflix = boto3.client('s3',
 region_name = REGION,
 aws_access_key_id = ACCESS_KEY_ID,
 aws_secret_access_key = SECRET_ACCESS_KEY)

responsenetflix=s3csvnetflix.put_object(Body=csv_buffernet.getvalue(),
                           Bucket=BUCKET_NAME,
                           Key=filenetflix)



userhulu = "hulu"
hulu = api.user_timeline(screen_name=userhulu,
                         count=200,
                         include_rts=True,
                         tweet_mode='extended')
huludf = pd.DataFrame()



for tweet in hulu:
    hashtags = []
    for hashtag in tweet.entities["hashtags"]:
        hashtags.append(hashtag["text"])
        text = api.get_status(id=tweet.id, tweet_mode='extended').full_text

    huludf = huludf.append(pd.DataFrame({'user_name': tweet.user.name,
                                           'user_location': tweet.user.location, \
                                           'user_description': tweet.user.description,
                                           'user_verified': tweet.user.verified,
                                           'date': tweet.created_at,
                                           'text': text,
                                           'hashtags': [hashtags if hashtags else None],
                                           're_tweet': tweet.retweet,
                                           'source': tweet.source}))
    huludf = huludf.reset_index(drop=True)

huludf["user_name"].replace({"Hoop Loop": "Hulu"}, inplace=True)

csv_bufferhulu=StringIO()
huludf.to_csv(csv_bufferhulu, index=False)
s3csvhulu = boto3.client('s3',
 region_name = REGION,
 aws_access_key_id = ACCESS_KEY_ID,
 aws_secret_access_key = SECRET_ACCESS_KEY)

responsehulu=s3csvhulu.put_object(Body=csv_bufferhulu.getvalue(),
                           Bucket=BUCKET_NAME,
                           Key=filehulu)


usertubi = "tubi"
tubi = api.user_timeline(screen_name=usertubi,
                         count=200,
                         include_rts=True,
                         tweet_mode='extended')
tubidf = pd.DataFrame()




for tweet in tubi:
    hashtags = []
    for hashtag in tweet.entities["hashtags"]:
        hashtags.append(hashtag["text"])
        text = api.get_status(id=tweet.id, tweet_mode='extended').full_text

    tubidf = tubidf.append(pd.DataFrame({'user_name': tweet.user.name,
                                           'user_location': tweet.user.location,
                                           'user_description': tweet.user.description,
                                           'user_verified': tweet.user.verified,
                                           'date': tweet.created_at,
                                           'text': text,
                                           'hashtags': [hashtags if hashtags else None],
                                           're_tweet': tweet.retweet,
                                           'source': tweet.source}))
    tubidf = tubidf.reset_index(drop=True)
csv_buffertubi=StringIO()
tubidf.to_csv(csv_buffertubi, index=False)

s3csvtubi = boto3.client('s3',
 region_name = REGION,
 aws_access_key_id = ACCESS_KEY_ID,
 aws_secret_access_key = SECRET_ACCESS_KEY)

responsetubi=s3csvtubi.put_object(Body=csv_buffertubi.getvalue(),
                           Bucket=BUCKET_NAME,
                           Key=filetubi)


netflix_path = 's3://{}:{}@{}/{}'.format(ACCESS_KEY_ID, SECRET_ACCESS_KEY, BUCKET_NAME, data_key_net)
netflixdf = pd.read_csv(smart_open(netflix_path))

hulu_path = 's3://{}:{}@{}/{}'.format(ACCESS_KEY_ID, SECRET_ACCESS_KEY, BUCKET_NAME, data_key_hul)
huludf = pd.read_csv(smart_open(hulu_path))

tubi_path = 's3://{}:{}@{}/{}'.format(ACCESS_KEY_ID, SECRET_ACCESS_KEY, BUCKET_NAME, data_key_tub)
tubidf = pd.read_csv(smart_open(tubi_path))

def preprocess(word):
    word = str(word)
    word = word.lower()
    word = word.replace('{html}', "")
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', word)
    rem_url = re.sub(r'http\S+', '', cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    return " ".join(filtered_words)

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

netflixdf = netflixdf.fillna({'hashtags': ' '})
huludf = huludf.fillna({'hashtags': ' '})
tubidf = tubidf.fillna({'hashtags': ' '})

netflixdf['english_text'] = netflixdf['text'].map(lambda s: preprocess(s))
huludf['english_text'] = huludf['text'].map(lambda s: preprocess(s))
tubidf['english_text'] = tubidf['text'].map(lambda s: preprocess(s))

netflixdf['english_text'] = netflixdf['english_text'].map(str)
huludf['english_text'] = huludf['english_text'].map(str)
tubidf['english_text'] = tubidf['english_text'].map(str)

spell = Speller(fast=True)
netflixdf['english_text'] = [' '.join([spell(i) for i in x.split()]) for x in netflixdf['english_text']]
huludf['english_text'] = [' '.join([spell(i) for i in x.split()]) for x in huludf['english_text']]
tubidf['english_text'] = [' '.join([spell(i) for i in x.split()]) for x in tubidf['english_text']]

netflixdf['words'] = netflixdf['english_text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
huludf['words'] = huludf['english_text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
tubidf['words'] = tubidf['english_text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

netflixdf['sentiment'] = netflixdf['words'].apply(lambda x: TextBlob(x).sentiment[0])
huludf['sentiment'] = huludf['words'].apply(lambda x: TextBlob(x).sentiment[0])
tubidf['sentiment'] = tubidf['words'].apply(lambda x: TextBlob(x).sentiment[0])



conditions = [
    (netflixdf['sentiment'] == 0.0),
    (netflixdf['sentiment'] <= -0.1),
    (netflixdf['sentiment'] >= -0.1)]

values = ['Netural', 'Negative', 'Positive', ]

netflixdf['results'] = np.select(conditions, values)




conditions = [
    (huludf['sentiment'] == 0.0),
    (huludf['sentiment'] <= -0.1),
    (huludf['sentiment'] >= -0.1)]

values = ['Netural', 'Negative', 'Positive', ]

huludf['results'] = np.select(conditions, values)




conditions = [
    (tubidf['sentiment'] == 0.0),
    (tubidf['sentiment'] <= -0.1),
    (tubidf['sentiment'] >= -0.1)]

values = ['Netural', 'Negative', 'Positive', ]
tubidf['results'] = np.select(conditions, values)


netflixdf['hashtagscloud'] = netflixdf['hashtags']
huludf['hashtagscloud'] = huludf['hashtags']
tubidf['hashtagscloud'] = tubidf['hashtags']
mediadata = pd.concat([netflixdf, huludf, tubidf], ignore_index=True)

twitterdata = mediadata[['user_name', 'date', 'english_text', 'words',  'sentiment', 'results',
     'hashtagscloud']]

twitterdata.loc[:,'dates'] = pd.to_datetime(twitterdata['date']).dt.date
twitterdata.loc[:,'times'] = pd.to_datetime(twitterdata['date']).dt.time
twitterdata.loc[:,'count'] = 1


csv_buffertwit = StringIO()
twitterdata.to_csv(csv_buffertwit, index=False)

s3csvcleandata = boto3.client('s3',
 region_name = REGION,
 aws_access_key_id = ACCESS_KEY_ID,
 aws_secret_access_key = SECRET_ACCESS_KEY)

responseclean=s3csvcleandata.put_object(Body=csv_buffertwit.getvalue(),
                           Bucket=BUCKET_NAME,
                           Key=cleandatame)


#NETFLIX
csv_buffernetclean=StringIO()
netflixdf.to_csv(csv_buffernetclean, index=False)

s3csvnetflixclean = boto3.client('s3',
 region_name = REGION,
 aws_access_key_id = ACCESS_KEY_ID,
 aws_secret_access_key = SECRET_ACCESS_KEY)

responsenetflixclean=s3csvnetflixclean.put_object(Body=csv_buffernetclean.getvalue(),
                           Bucket=BUCKET_NAME,
                           Key=filenetflixclean)

#HULU
csv_bufferhuluclean=StringIO()
huludf.to_csv(csv_bufferhuluclean, index=False)

s3csvhuluclean = boto3.client('s3',
 region_name = REGION,
 aws_access_key_id = ACCESS_KEY_ID,
 aws_secret_access_key = SECRET_ACCESS_KEY)

responsehuluclean=s3csvhuluclean.put_object(Body=csv_bufferhuluclean.getvalue(),
                           Bucket=BUCKET_NAME,
                           Key=filehuluclean)

#TUBI
csv_buffertubiclean=StringIO()
tubidf.to_csv(csv_buffertubiclean, index=False)

s3csvtubiclean = boto3.client('s3',
 region_name = REGION,
 aws_access_key_id = ACCESS_KEY_ID,
 aws_secret_access_key = SECRET_ACCESS_KEY)

responsetubiclean=s3csvtubiclean.put_object(Body=csv_buffertubiclean.getvalue(),
                           Bucket=BUCKET_NAME,
                           Key=filetubiclean)

