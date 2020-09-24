'''
Collecting data from twitter
'''

import numpy as np
import tweepy
import json
import panda as pd
from tweepy import OAuthHandler

# Credentials
consumer_key = "DmrnOi7TveygMUYwulmTkfIIk"
consumer_secret = "HINOf63NINtsFGngCfF8k4pGLMJyyMG6BXP5NabsVAAd3eatAx"
bearer_token = "AAAAAAAAAAAAAAAAAAAAAIPkHwEAAAAAbed5E7po6C4I8uF1gDQRc5m6jI0%3Dj3zgPCjrzwG6Wijb6hAYUKIiv9c7713UR3DyYNbBo5pmqw7Rum"
access_token = "517219863-7dv5B0JAACA8BNi4RIidQ0wk7jPC1nq36bO8ApGL"
access_token_secret = "ur1ZDGbksZSAt0h34hxiKwLprMo5OEv6b3IKin1QLakMR"

# Calling API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# provide the query you want to pull the data
query = "motorolla"

# Fetching tweets
Tweets = api.search(query, count=10, lang='en',
                    exclude='retweets', tweet_mode='extended')

