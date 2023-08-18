# Sentiment-Analysis-
This project focuses on the application of Natural Language Processing (NLP) techniques, particularly sentiment analysis, to a sizable dataset of customer feedback from Airbnb. The goal is to gain insights into customer behavior and predict sentiments using textual data.

**Code Flow:**

1) **Setup:**
Connection to Google Colab.
Installation of PySpark library on Google Colab.
2) **Data Loading:**
Loading the Airbnb customer feedback dataset.
3) **Importing Libraries and Creating Spark Object:**
Importing necessary libraries and packages for data processing and analysis.
Creating a Spark object for efficient distributed processing.
4) **Text Pre-Processing:**
Utilizing User-Defined Functions (UDFs) for various text processing tasks:
  - Converting text to lowercase.
  - Removing non-ASCII characters.
  - Handling abbreviations.
  - Eliminating specific features.
  - Removing punctuation and alphanumeric words.
  - Removing stop words.
5) **Part-of-Speech (POS) Tagging:**
Implementing UDF functions to perform POS tagging on the pre-processed text.
6) **Text Lemmatization:**
Applying lemmatization, a technique in NLP that reduces words to their base or root forms, aiding in semantic analysis and similarity identification.
7) **Sentiment Analysis:**
Leveraging the textblob library to assign sentiment polarity to each customer review.
Creating a binary rating as the target variable: 1 for positive sentiment and 0 for negative sentiment.
8) **Visualization:**
Using the matplotlib library to visualize the distribution of sentiments within the dataset.

*********************************************************************************************************

**Text Pre-Processing:**
a. A series of UDFs are designed and implemented for text pre-processing tasks, including lowercasing, non-ASCII removal, abbreviation resolution, feature elimination, punctuation and alphanumeric word removal, and stop word removal.

b. POS tagging is performed using UDF functions, enhancing linguistic analysis.

c. Text lemmatization is applied to reduce words to their root forms for improved semantic understanding.

**Sentiment Analysis:**
The textblob library is employed to assign sentiment polarity (positive, negative, neutral) to each customer review. A binary rating is also generated, categorizing reviews as either positive (1) or negative (0).

**Visualization:**
Matplotlib is utilized to create visual representations of the sentiment distribution, providing insights into the overall sentiment trends within the Airbnb customer feedback dataset.

In conclusion, this project employs NLP techniques, including text pre-processing and sentiment analysis, to extract valuable insights from Airbnb customer feedback data, enabling the prediction and visualization of customer sentiments and behaviors.







C:\Users\Arpita\Downloads\Sentiment Analysis
