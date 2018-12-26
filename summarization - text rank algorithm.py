import nltk
nltk.download('stopwords')

from bs4 import BeautifulSoup
from urllib.request import urlopen  
import re     
import nltk
page = urlopen('https://en.wikipedia.org/wiki/Automatic_summarization')
soup = BeautifulSoup(page)
doc_list=[]
for i in soup.find_all('p'):
    doc_list.append(i.text)
doc=' '.join(doc_list)

sentence_list = nltk.sent_tokenize(doc)

from sklearn.feature_extraction.text import CountVectorizer
c = CountVectorizer()
bow_matrix = c.fit_transform(sentence_list)
bow_matrix

from sklearn.feature_extraction.text import TfidfTransformer
normalized_matrix = TfidfTransformer().fit_transform(bow_matrix)

similarity_graph = normalized_matrix * normalized_matrix.T
similarity_graph.toarray()

import networkx as nx
nx_graph = nx.from_scipy_sparse_matrix(similarity_graph)
scores = nx.pagerank(nx_graph)

ranked = sorted(((scores[i],s) for i,s in enumerate(sentence_list)),reverse=True)
for i in range(7):
    print(ranked[i][1])
