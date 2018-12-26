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

from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords

print(summarize(doc, ratio=0.05))
print(keywords(doc, ratio=0.01))
