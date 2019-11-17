"""
main_program.py
~~~~~~~~~~~~~
This executes the automatic document clustering by extracting, translating, cleaning and clustering
"""

# Package Loading
import os
import pandas as pd
from matplotlib import pyplot as plt
import time
import pickle5 as pickle

import warnings
warnings.filterwarnings("ignore")

# Setting current directory
answer = input('Set current working directory to main folder (subfolders: program & data & results)  \n (Example: /Users/konradeilers/Documents/01_Studies/04_Practice/keilers) ')
folder_path = answer.strip()
os.chdir(answer.strip())

path_to_program = folder_path + '/program'
path_to_data = folder_path  + '/data'
path_to_result = folder_path  + '/results'

os.chdir(path_to_program)

# Importing modules
import extraction
import language
import input_
# Importing classes
import clustering

# Data
data_method = input_.data_method()
if data_method == 'y': #asking the user for the data generation method
    # Extraction from Wikipedia
    n = 200
    languages = ["en", "fr", "de"]
    data = extraction.wiki_extraction(n, languages)
    
    #os.chdir(path_to_data)
    #data.to_csv('wiki_extraction.csv')
elif data_method == 'n':
    # Loading pre-loaded data into the environment
    os.chdir(path_to_data)
    data = pd.read_csv('wiki_extraction.csv')
    os.chdir(path_to_program)

# Presuming we were not aware of the extraction method
# Language detection
data['Language_predict'] = language.language_predictor(data, "Text")
data['Language_predict_full'] = language.full_language(data, "Language_predict")
# Selecting relevant columns for clustering
data = data.loc[:,['Title','Text', 'Language_predict', 'Language_predict_full']]

# Remove probable mistakes in language detection
print('As we are clustering by language, languages with less than 10 occurances are removed')
mistake_occurances = data.Language_predict_full.value_counts() < 10
remove_index = []
for i in mistake_occurances[mistake_occurances == True].index.tolist():
    print(f"Languages to be removed from dataset: {i}")
    remove_index.append(data[data.Language_predict_full == i].index.tolist())
print(f"A total of {len([item for elem in remove_index for item in elem])} entries will be removed")
data = data.drop(data.index[[item for elem in remove_index for item in elem]])
data = data.reset_index()
print(f"The new dataset has now {data.shape[0]} number of rows")

# Remove languages as not supported by nltk corpus
#Sample answer: /Users/konradeilers/nltk_data/corpora/stopwords
running_nltk_corpus = input_.use_nltk_corpora() #asking the user for usage of nltk corpus
if len(running_nltk_corpus) > 0:
    lister = os.listdir(str(running_nltk_corpus))
    print(f"languages supported in nltk corpus: {lister}")
    data['supported'] = language.remove_no_support(data, 'Language_predict_full', running_nltk_corpus)
    print(f'Number of entries to be removed: {data[data.supported == 0].shape[0]}')
    data = data[data.supported == 1]
    data = data.reset_index()
    print(f'The new dataset has {data.shape[0]} rows')

# Pre-Processing
data['Text'] = language.remove_white_space(data,"Text") # Remove whitespace & lower everything
data['Text'] = language.pre_text_tokenizer(data,"Text") # Tokenize
data['Text'] = language.remove_single_words(data,"Text") # Remove single letter words 
data['Text'] = language.remove_non_words(data,"Text") # Remove non-words

if len(running_nltk_corpus) > 0:
    data['Text'] = language.stopwords_remover(data, "Text",'Language_predict_full') # Remove stopwords each language
    data['Text'] = language.stemmer(data, "Text", 'Language_predict_full') # Stemming in each language

# Embedding & Clustering
os.chdir(path_to_result)

max_number_clusters = input_.number_of_max_clusters() # asking the user for maximum number of cluster
fig_1 = "PCA_plot"
fig_2 = "Hierarchical_plot"
final_result = {}
for i in data['Language_predict'].unique():
    subdata = data.loc[data['Language_predict'] == i]
    model_cluster = clustering.VecFormat(data = [" ".join(i) for i in subdata['Text']], max_df=0.8, max_features=None, ngram_range=(1,1))
    # printing top words
    print(f'Printing the top 10 words per cluster for language: {i}')
    model_cluster.top_words(max_number_clusters,10)
    time.sleep(2)
    # assigning labels
    labels = model_cluster.k_means_optimal_cluster(max_number_clusters)['labels']
    subdata['Labels'] = labels
   
    # visualizations
    picture_name_1 = fig_1 + "_" + str(i)
    model_cluster.apply_2D_PCA_visualize(max_number_clusters, picture_name_1)
    
    picture_name_2 = fig_2 + "_" + str(i)
    model_cluster.hierarchical_clustering_visualize(list(subdata.Title))
    plt.savefig(str(picture_name_2))
    
    # final assignment into dictionary
    sub_dict = {i : subdata}
    final_result.update(sub_dict)


with open('data.p', 'wb') as fp:
    pickle.dump(final_result, fp, protocol=pickle.HIGHEST_PROTOCOL)

"""
with open('data.p', 'rb') as fp:
    data = pickle.load(fp)
"""