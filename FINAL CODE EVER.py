# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 14:56:15 2023

@author: J a c y
"""

# Import the excel and read it
# pandas libraries help to read the excel, string libraries help to remove the punctuation
# usecols reads only the next two rows, it doesn't include the time stamp
import pandas as pd
import string

data = pd.read_excel("C:/Users/J a c y/Desktop/Final week!/Top FAQs of an e-commerce website (Responses).xlsx", usecols=[1, 2])

def remove_punctuation(text):
    punctuation = set(string.punctuation) - set('/')
    translator = str.maketrans("", "", ''.join(punctuation))
    return text.translate(translator)

# Remove punctuation from a specific column
columns_to_clean = ['Select TOP 3 questions that you may have when you shop online (food, products, clothing, technology etc.)', 'Are there any other questions that you usually have when you shop online?']
data[columns_to_clean] = data[columns_to_clean].applymap(remove_punctuation)

# Print the modified DataFrame
print(data)

# removing stop words
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Assuming your DataFrame is named 'df' and the columns containing text data are named 'text_column1' and 'text_column2'

# Download the stop words corpus if you haven't done so already
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')

# Get the list of stop words
stop_words = set(stopwords.words('english'))

# Define a function to remove stop words from a text
def remove_stop_words(text):
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words or token == '/']
    return ' '.join(filtered_tokens)

# Apply the function to the 'text_column1' in the DataFrame
data['Select TOP 3 questions that you may have when you shop online (food, products, clothing, technology etc.)'] = data['Select TOP 3 questions that you may have when you shop online (food, products, clothing, technology etc.)'].apply(remove_stop_words)

# Apply the function to the 'text_column2' in the DataFrame
data['Are there any other questions that you usually have when you shop online?'] = data['Are there any other questions that you usually have when you shop online?'].apply(remove_stop_words)

print(data)

from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('wordnet')

data = pd.read_excel("C:/Users/J a c y/Desktop/Final week!/Top FAQs of an e-commerce website (Responses).xlsx", usecols=[1, 2])

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def remove_punctuation(text):
    punctuation = set(string.punctuation) - set('/')
    translator = str.maketrans("", "", ''.join(punctuation))
    return text.translate(translator)

def stem_and_lemmatize(text):
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in stemmed_tokens]
    return ' '.join(lemmatized_tokens)

# Remove punctuation from a specific column
columns_to_clean = ['Select TOP 3 questions that you may have when you shop online (food, products, clothing, technology etc.)', 'Are there any other questions that you usually have when you shop online?']
data[columns_to_clean] = data[columns_to_clean].applymap(remove_punctuation)

# Stem and lemmatize the cleaned text in the specific columns
data[columns_to_clean] = data[columns_to_clean].applymap(stem_and_lemmatize)

# Print the modified DataFrame
print(data)
