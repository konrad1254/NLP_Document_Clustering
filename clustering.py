#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clustering.py
~~~~~~~~~~~~~
The class VecFormat introduces a class that uses a TfidfVectorizer to generate a sparse matrix
Then, the class covers clustering and visualizatiion
"""

# Import
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from scipy.cluster.hierarchy import ward, dendrogram

from plotnine import *
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class VecFormat:
    def __init__(self, data,max_df,max_features,ngram_range):
        self.data = data
        self.max_df = max_df
        self.max_features = max_features
        self.ngram_range = ngram_range
    
    def tfidf_vectorize(self):
        """
        Creating a tf-idf sparse matrix
        """
        tfidf_vectorizer = TfidfVectorizer(self.max_df,self.max_features, self.ngram_range)
        return tfidf_vectorizer.fit_transform(self.data)
    
    def get_feature_names(self):
        """
        Getting feature names
        """
        tfidf_vectorizer = TfidfVectorizer(self.max_df,self.max_features, self.ngram_range)
        tfidf_vectorizer.fit_transform(self.data)
        return tfidf_vectorizer.get_feature_names()
    
    def to_dense(self):
        """
        Return a dense matrix representation
        """
        tfidf_vectorizer = TfidfVectorizer(self.max_df,self.max_features, self.ngram_range)
        r = tfidf_vectorizer.fit_transform(self.data)
        return r.todense()
    
    def dist(self):
        """
        Dist defined as 1 - the cosine similarity of each document. Generate a measure of similarity between docs.
        """        
        tfidf_vectorizer = TfidfVectorizer(self.max_df,self.max_features, self.ngram_range)
        r = tfidf_vectorizer.fit_transform(self.data)
        dist = 1 - cosine_similarity(r)
        return dist
        
    def tfidf_kmeans(self,  n_clusters):
        """
        Performing k-means clustering on the sparse matrix
        """
        sparse_matrix = self.tfidf_vectorize()
        km = KMeans(n_clusters)
        km.fit(sparse_matrix)
        km_results = {"labels": km.labels_,
                      "silhouette_score": silhouette_score(sparse_matrix, km.labels_),
                      "centroids": km.cluster_centers_}
        return km_results
    
    def silhouette_score(self, max_n_clusters):
        """
        Obtains the silhouette_score as a function of the maximum number of clusters as input
        """
        list_l = list() 
        list_i = list()
        for i in range(2,max_n_clusters+1):
            km_save = self.tfidf_kmeans(i)
            list_l.append(km_save['silhouette_score'])
            list_i.append(i)
        return list(zip(list_i,list_l))
    
    def silhouette_score_plot(self, max_n_clusters):
        """
        Plotting the silhouette score for max_n_clusters clusters
        """  
        to_plot = pd.DataFrame(self.silhouette_score(max_n_clusters), columns = ["n_clusters", "Silhouette_score"])
        return ggplot(aes(x='n_clusters', y='Silhouette_score'), to_plot) + geom_line() + geom_point(color = "red")
    
    def optimal_number_cluster(self,max_n_clusters):
        """
        Determines the best # of clusters given the range of clusters
        """
        silhouette_score_list = self.silhouette_score(max_n_clusters)
        max_sil = max([silhouette_score_list[i][1] for i in range(len(silhouette_score_list))])
        return [silhouette_score_list[i][0] for i in range(len(silhouette_score_list)) if silhouette_score_list[i][1] == max_sil]
    
    def k_means_optimal_cluster(self, max_n_clusters):
        """
        K-Means cluster with optimal clusters
        """
        optimal_cluster = self.optimal_number_cluster(max_n_clusters)[0]
        return self.tfidf_kmeans(optimal_cluster)
    
    def top_words(self, max_n_clusters, n_terms):
        """
        Print top words per cluster
        """
        k_means_result_labels = self.k_means_optimal_cluster(max_n_clusters)['labels']
        words = self.get_feature_names()

        df = pd.DataFrame(self.to_dense()).groupby(k_means_result_labels).mean()
        
        for i,r in df.iterrows():
            print('\nCluster {}'.format(i))
            print(','.join([words[t] for t in np.argsort(r)[-n_terms:]]))
    
    def apply_2D_PCA_visualize(self,max_n_clusters, name):
        """
        Applying PCA for visualisation. Selecting subset
        """    
        text = self.to_dense()
        labels = self.k_means_optimal_cluster(max_n_clusters)['labels']        
        pca = PCA(n_components=2).fit_transform(text)
        
        fig = plt.figure()
        plt.scatter(pca[:, 0], pca[:, 1], c = labels)
        plt.title('PCA Cluster Plot')
        
        return fig.savefig(str(name))
    
    def hierarchical_clustering_visualize(self, titles): 
        """
        Hierarchical visualization
        """
        linkage_matrix = ward(self.dist())
        
        fig, ax = plt.subplots(figsize=(15, 20)) # set size
        ax = dendrogram(linkage_matrix, orientation="right", labels=titles); #
        
        plt.tick_params(axis= 'x',which='both',bottom='off',top='off',labelbottom='off')
        
        return plt.tight_layout()