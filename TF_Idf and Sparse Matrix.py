#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import sys
import os


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


df=pd.read_csv('https://raw.githubusercontent.com/Anu0603/Twitter_data/main/twitter30k_cleaned.csv')


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df['sentiment'].value_counts()


# In[ ]:


x=df['twitts']
y=df['sentiment']


# In[ ]:


tfidf=TfidfVectorizer()
x=tfidf.fit_transform(x)


# In[ ]:


x


# In[ ]:


x.shape, y.shape


# In[ ]:


type(x),type(y)


# In[ ]:


sys.getsizeof(x)


# In[ ]:


sys.getsizeof('hello')


# In[ ]:


print(tfidf.vocabulary_)


# In[ ]:


d=(x.data.nbytes+ x.indptr.nbytes+x.indices.nbytes)


# In[ ]:


d


# In[ ]:


d/2**20


# In[ ]:


x.shape


# In[ ]:


(x.shape[0]*x.shape[1])*8/2**20


# Non-Negative Matrix factorization

# In[ ]:


from sklearn.decomposition import NMF
x_train,x_test, y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0, stratify=y)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'nmf=NMF(n_components=500, random_state=0)\nx_train_nmf=nmf.fit_transform(x_train)')


# In[ ]:


x_train.shape, x_train_nmf.shape


# In[ ]:


def run_svm(clf,x_test,x_train,y_train,y_test):
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    print()
    print('Printing Report')
    print('classification_report(y_test,y_pred)')
    


# In[ ]:


get_ipython().run_cell_magic('time', '', 'clf=LinearSVC()\nx_test_nmf = nmf.transform(x_test)\n\nrun_svm(clf,x_train_nmf,x_test_nmf,y_train,y_test)')


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0, stratify = y)
%%time
clf = LinearSVC()
run_svm(clf, x_train, x_test, y_train, y_test)


# In[ ]:





# In[ ]:


## Truncated Singular Value Decomposition (TSVD)


# In[ ]:


from sklearn.decomposition import TruncatedSVD as TSVD


# In[ ]:


get_ipython().run_cell_magic('time', '', 'tsvd = TSVD(n_components=500,random_state=0)\nx_train_tsvd = tsvd.fit_transform(x_train)\n')


# In[ ]:


sum(tsvd.explained_variance_)


# In[ ]:


x_train.shape, x_train_nmf.shape


# In[ ]:


x_test_tsvd=tsvd.transform(x_test)

x.shape, x_train_tsvd.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n\nclf = LinearSVC()\nrun_svm(clf, x_train_tsvd, x_test_tsvd, y_train, y_test)')


# In[ ]:


x_train_tsvd


# In[ ]:


(x.shape[0] * x.shape[1])*8/(2**20)


# In[ ]:


sys.getsizeof(x_train_tsvd)/(2**20) 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




