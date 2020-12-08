# NLP Document Clustering

## Author: Konrad Eilers
## Date: 16/Nov/2019

Description: Automatic Document Clustering Algorithm 

How to run the program:
- Executing main_program.py in python

Inputs required from user:
- Current working directory: The user will be asked to set the current working directory to the main folder. On my computer, for example, this would be: 
```/Users/konradeilers/Documents/01_Studies/04_Practice/keilers```
- Folder directory of nltk corpora stopwords data. This is required for stop words & stemming and for eliminating languages that are not supported by nltk. On my computer, this would be: ```/Users/konradeilers/nltk_data/corpora/stopwords```

Options to the user:
- The user can decide to run the webscraper or work with a pre-loaded dataset. As the wikipedia API is quite slow, the web scraper will take some time to run (ca. 10 min on my computer)
- The user can decide to run the stop words & stemming process with nltk. If selected, this will cause the languages that are not supported by nltk to be excluded from the dataset 

Structure of code:
- Main Program:
	- This executes the automatic document clustering by extracting, translating, cleaning and clustering
- Extraction module:
	- The extraction method uses the wikipedia API to generate a random search. If there multiple entries for one search, a new random search gets started
	- Currently, the extraction method searches for 200 articles in English, German and French each. However, this can be changed in the main_program
	- The code only extracts the summaries for simplicity
- Language module:
	- Module that includes all the functions that are relevant for language detection and text processing
	- Notice that the langdetect functions (detection of languages) functions much better in European languages than others
- Clustering class:
	- The class VecFormat introduces a class that uses a TfidfVectorizer to generate a sparse matrix
	- On this basis, the class includes the clustering and visualisations
- input_ module:
	- user inputs are defined here

Folder Structure:
	- data: pre-loaded dataset
	- program: all required programs / modules
	- results: empty folder where results will be saved
		
Important notes / assumptions:
	- Code requires stable internet connection
	- I am using Python 3.3+. Hence,  __init__.py is not required
	- For improved readability, warnings are not printed
	- Wikipedia extraction: The code extracts 200 summaries per language to fulfil the requirements of the exercise. In the next step, however, the code
	"forgets" the true labels and uses langdetect to detect the language of the document
	- Language detection: 
		- In order to perform well in clustering, the code removes data from languages with less than 10 occurrences per language.
		- Looking at multilingual document clustering research (DOI: 10.3115/v1/D14-1065), I believe the best method would be to translate all different languages back to English. This would capture the fact that documents may be about the same thing in different languages. However, I couldn't find a free, stable translation API that allowed for such services (TextBlob & Google Translate (free version) both broke down).
	- TfidfVectorizer:
		- I chose this simple NLP over more complex word embeddings (i.e. Word2Vec) for two reasons. First, this method is less memory intensive. As I am working on my own computer, this is an important criteria. Second, I am less interested in understanding the relationship between each word, but more in the overall context of the document. For this, I believe looking at the inverse document frequency should be sufficient.
	- Method of clustering
		- There is a plethora of different clustering algorithms (K-Means, Affinity Propagation, Mean Shift,...)
		- This algorithm runs on K-means. There are obvious disadvantage (mainly we need to specify the number of clusters). However, it is very efficient and that is why I chose it
	- Method of deciding the optimal number of clusters: based on Silhouette Score (https://en.wikipedia.org/wiki/Silhouette_(clustering))
	
Output: 
	- Pickle file: Dictionary with DataFrame per language. The dictionary includes Title, Text, Language_predict and Labels as columns
	- PCA visualisation: image of labelled scatter plot in 2 dimensions 
	- Hierarchical clustering visualisation 

Conclusion:
	- Given the randomisation underlying the extraction method, there will be inherently a low relationship between the documents
	- Given that the algorithm does not translate back to English, the code does not pick up relationships between different languages
	- In the next iteration, one should 
		- replace the stemming method by lemmatisation (more accurate way of generating the root of a word based on parts-of-speech)
		- Re-run the analysis with translated documents (via official google translate API: https://cloud.google.com/translate/docs/)
		- Introduce word embedding (i.e. Word2Vec)
		- Introduce different clustering mechanism (such as Affinity Propagation), where the user does not have to decide on the maximum of clusters	 

Main reference articles:
	- http://brandonrose.org/clustering
	- https://www.kaggle.com/jbencina/clustering-documents-with-tfidf-and-kmeans 
	- https://medium.com/@dcameronsteinke/tf-idf-vs-word-embedding-a-comparison-and-code-tutorial-5ba341379ab0

    

