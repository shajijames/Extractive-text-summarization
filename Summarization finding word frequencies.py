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

clean_doc = str.lower(re.sub(r'(\[[0-9]*\])|(\s+)|([^a-zA-Z])', ' ', doc))

stopwords = nltk.corpus.stopwords.words('english')
word_frequencies = {}  
for word in nltk.word_tokenize(clean_doc):  
    if word not in stopwords:
        if word not in word_frequencies.keys():
            word_frequencies[word] = 1
        else:
            word_frequencies[word] += 1

maximum_frequncy = max(word_frequencies.values())

for word in word_frequencies.keys():  
    word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
    
sentence_scores = {}  
for sent in sentence_list: 
    for word in nltk.word_tokenize(sent.lower()):
        if word in word_frequencies.keys():
            if len(sent.split(' ')) < 30:
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word]
                else:
                    sentence_scores[sent] += word_frequencies[word]

import heapq  
summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

summary = ' '.join(summary_sentences)  
print(summary) 
