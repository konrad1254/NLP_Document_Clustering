#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
input_.py
~~~~~~~~~~~~~
User inputs are defined here
"""

def data_method():
    print('You can either run the webscraper (which takes time) or run a pre-loaded dataset')
    answer = input("Run webscraper [y/n]:")
    while answer not in ["y", "n"]:
        answer = input("Answer not in answer choices. Run webscraper [y/n]:")
    return answer.lower().strip()

def use_nltk_corpora():
    print('In order to use stemming & stopwords remover, the program requires the filepath where it can find the nltk_corpora stopwords. In this step, languages are removed that are not supported by nltk')
    answer = input("Use stemming & stopwords remover [y/n]:")
    while answer not in ["y", "n"]:
        answer = input("Answer not in answer choices. Use stemming & stopwords remover[y/n]:")

    if answer.lower().strip() == 'y':
        filepath = input("Full filepath where the nltk_corpora for stopwords is saved (as path not as string. For example: /Users/konradeilers/nltk_data/corpora/stopwords):")
        return filepath.lower().strip()
    if answer.lower().strip() == 'n':
        return []

def number_of_max_clusters():
    print('Lets move to the clustering. What is the maximum number of clusters you want to consider?')
    answer = int(input("Maximum # of clusters:"))
    return answer
    