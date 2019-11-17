"""
language.py
~~~~~~~~~~~~~
Module that includes all the functions that are relevant for language detection and text processing
"""

# Import
from langdetect import detect
import pycountry
import re
import os

from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

import warnings
warnings.filterwarnings("ignore")

# Module for language cleaning and detection
def language_predictor(data,column_with_text):
    """
    Detects language of text in ISO 639-1 format
    """
    return [*(map(detect,list(data[str(column_with_text)])))]

def full_language(data, column_with_language):     
    """
    Generates full language name
    """
    return data[str(column_with_language)].apply(lambda x: pycountry.languages.get(alpha_2=x).name)

def support_score(language, filepath):
    """
    Check if language is supported by corpora
    """
    if language.lower().strip() in os.listdir(str(filepath)):
        return 1
    else:
        return 0

def remove_no_support(data, column_with_language, filepath):
    """
    Remove unsupported languages
    """
    return data[str(column_with_language)].apply(lambda x: support_score(x, filepath))

def remove_white_space(data, column_with_text):
    """
    Removes white space & lowers text
    """
    return data[str(column_with_text)].apply(lambda x: x.strip().lower())

def pre_text_tokenizer(data, column_with_text):
    """
    Tokenizes text
    """
    return data[str(column_with_text)].apply(lambda x: wordpunct_tokenize(x))

def longer_than(coulmn_entry, n = 1):
    """
    Removes words that are just 1 character long
    """
    return [i for i in coulmn_entry if len(i) > 1]

def remove_single_words(data, column_with_text):
    """
    Applies longer_than function to column
    """
    return data[str(column_with_text)].apply(lambda x: longer_than(x))

def regexer(regex,coulmn_entry):
    """
    Finds all regex patterns in a column that are different from 0
    """
    return [i for i in coulmn_entry if len(re.findall(regex, i)) != 0]

def remove_non_words(data, column_with_text, regex = "^[a-zA-Z]"):
    """
    Applies regexer function to a column. Speficing regex such that it only captures strings that start with letters
    """
    return data[str(column_with_text)].apply(lambda x: regexer(regex,x))

def stopwords_f(word_list, language):
    """
    Excludes stopwords depending on word list
    """
    stop_words = set(stopwords.words(str(language))) 
    return [w for w in word_list if not w in stop_words]
    
def stopwords_remover(data,column_with_text,column_with_language):
    """
    Applies stopwords_f to column depending on language
    """
    result = []
    for i in range(len(data)): 
        result.append(stopwords_f(data[str(column_with_text)][i], data[str(column_with_language)][i]))
    return result

def snowball_stemmer(word_list, language):
    """
    Generating stemmer based on language
    """
    stemmer = SnowballStemmer(str(language).lower())
    return [stemmer.stem(item) for item in word_list]

def stemmer(data,column_with_text, column_with_language):
    """
    Applying stemmer function depending on language
    """
    result = []
    for i in range(len(data)):
        result.append(snowball_stemmer(data[str(column_with_text)][i],data[str(column_with_language)][i] ))
    return result