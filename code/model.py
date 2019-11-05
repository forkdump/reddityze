#Importing necessary libraries
import math
import string
import praw
import pandas as pd
import numpy as np
import emoji
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
#nltk.download('stopwords')
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
#nltk.download('vader_lexicon')
from pprint import pprint

#PRAW Configuration
reddit = praw.Reddit(client_id='StmWjOzTODd8Og', client_secret='3-HWmn6SD7a32SEoOLXyVSUv5jQ', user_agent='r/webscraper')

#Global variables
comments_record = []

#All functions
def fetch_posts_and_comments(subreddit):
  posts = []
  #Fetching 500 hottest post titles and corresponding comments from the entered subreddit
  for post in reddit.subreddit(subreddit).hot(limit=250):
    posts.append(post.title)
    submission = reddit.submission(id=post.id)
    submission.comments.replace_more(limit=0)
    new_comments = set(list(map(lambda comment:comment.body, submission.comments)))
    global comments_record
    comments_record.append(preprocessing(new_comments))
  return posts

def preprocessing(posts):
  remove = set(stopwords.words('english'))
  filtered = []
  for post in posts:
    #Converting numbers to words, translation of emojis and removing punctuations to give bag of words
    post = [emoji.demojize(x, delimiters=('.', '.')) for x in post.translate(str.maketrans(dict.fromkeys(string.punctuation))).replace('“', '').replace('”', '').replace('‘', '').replace('’', '').lower().split()]
    #Removing stopwords and words irrelevant to sentiment, eliminating quotes
    temp = [x for x in post if x not in remove]
    filtered.append(" ".join(temp).strip())
  return filtered

def sentiment_analysis(filtered):
  sia = SIA()
  results = []
  for post in filtered:
    #Results of sentiment analysis
    score = sia.polarity_scores(post)
    score['post'] = post
    results.append(score)
  data = pd.DataFrame.from_records(results)
  return data

def labelling(data, threshold):
  #Labelling the sentiments depending on a compound threshold(can be adjusted)
  data['label'] = 0
  data.loc[data['compound'] > threshold, 'label'] = 1
  data.loc[data['compound'] < (threshold * -1), 'label'] = -1
  return data

def top_posts(data):
  #Sample posts from positive and negative label categories
  results = []
  results.append(list(data[data['label'] == 1].post))
  results.append(list(data[data['label'] == 0].post))
  results.append(list(data[data['label'] == -1].post))
  return results

top = top_posts(labelling(sentiment_analysis(preprocessing(fetch_posts_and_comments('music'))), 0.15))
print('Positive Posts- ' +str(len(top[0])))
print('Neutral Posts- ' +str(len(top[1])))
print('Negative Posts- ' +str(len(top[2])))
