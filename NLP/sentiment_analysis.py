#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Modify Time 2023/8/6 00:02
@Author maoalong  
@Version 1.0
@Desciption None
"""
from textblob import TextBlob


def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity

    if sentiment <= 0:
        # 消极
        sentiment_type = 0
    elif sentiment > 0:
        # 积极
        sentiment_type = 4
    return sentiment_type, sentiment
