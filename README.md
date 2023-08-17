# Sentiment-Analysis-
Used text pre processing and NLP techniques on AIR BNB customer feedback data for sentiment analysis

This project uses large AIR BNB customer feedback dataset and applies Natural Language Processing techniques like sentiment analysis to predict customer behavior.

Code Flow
Connect to Google Collab  
Install PySpark on Google Collab  
Load Dataset  
Importing libraries and packages  
Create Spark object  
Pre-proccessing Data  
Sentiments Analysis  
Predictions  
Visualization

Text Pre-Processing #a. Created UDF Functions for text processing: Convert to lower case, Remove nonAscii, Fix abbreviations, remove features, remove Punctuations and alphanumeric words, Remove Stop words

#b. POS tagging using UDF functions

#c. Text Lemmatization: Lemmatization is a text pre-processing technique used in natural language processing (NLP) models to break a word down to its root meaning to identify similarities.

Sentiment Anallysis Utilized textblob function to assign sentiment polarity to each customer review. Assigned Binary Rating as Target Variable 1: Positive 0: Negative

Visualization used matplotlib to analyse sentiment distribution.

C:\Users\Arpita\Downloads\Sentiment Analysis
