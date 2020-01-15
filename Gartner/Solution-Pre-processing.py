import numpy as np
import pandas as pd

#for text processing
import re
import string
import nltk 
from nltk.corpus import stopwords
from textblob import Word

#calculation of time
from time import time

##pretty print
import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

from gensim.corpora import Dictionary

# Build LDA model
from gensim.models.ldamulticore import LdaMulticore

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# spacy 
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('max_colwidth', -1)


# #### Merge all 3 sheets

# In[2]:


# from pandas import ExcelWriter
# from pandas import ExcelFile

xls = pd.ExcelFile('data.xlsx')
df1 = pd.read_excel(xls, sheet_name='Aug')
df2 = pd.read_excel(xls, sheet_name='Sept')
df3 = pd.read_excel(xls, sheet_name='Oct')


# In[3]:


df = pd.concat([df1,df2,df3] , ignore_index=True)


# ## Inspect Text field

# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


#fetch missing values of a column

df[df["Query Text"].isnull()]


# In[8]:


#drop all the rows which have NaN in Query Text

df = df.dropna(how='any',axis=0) 


# In[9]:


df.isnull().sum()


# In[10]:


df.drop_duplicates(subset ="Query Text", 
                     keep = 'last', inplace = True) 


# In[11]:


df.info()


# In[12]:


# check the length of documents

document_lengths = np.array(list(map(len, df['Query Text'].str.split(' '))))

print("The average number of words in a document is: {}.".format(np.mean(document_lengths)))
print("The minimum number of words in a document is: {}.".format(min(document_lengths)))
print("The maximum number of words in a document is: {}.".format(max(document_lengths)))


# In[13]:


print("There are {} documents with tops 5 words.".format(sum(document_lengths == 1)))
print("There are {} documents with tops 5 words.".format(sum(document_lengths == 2)))
print("There are {} documents with tops 5 words.".format(sum(document_lengths == 3)))
print("There are {} documents with tops 5 words.".format(sum(document_lengths == 4)))
print("There are {} documents with tops 5 words.".format(sum(document_lengths == 5)))


# ## Task 1
#   
# ### Sub-task 2 : Text pre-processing

# In[14]:


def text_preprocessing(data):
    
    #convert text to lower-case
    data['processed_text'] = data['Query Text'].apply(lambda x:' '.join(x.lower() for x in x.split()))

    #remove punctuations, unwanted characters
    data['processed_text_1']= data['processed_text'].apply(lambda x: "".join([char for char in x if char not in string.punctuation]))

    #remove numbers
    data['processed_text_2']= data['processed_text_1'].apply(lambda x: re.sub('[0-9]+', ' ' , x))

    #remove stopwords
    stop = stopwords.words('english')
    data['processed_text_3']= data['processed_text_2'].apply(lambda x:' '.join(x for x in x.split() if not x in stop))

    #lemmatization
    data['processed_text_4']= data['processed_text_3'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    # remove all single characters
    data['processed_text_5'] = data['processed_text_4'].apply(lambda x: re.sub(r'\s+[a-zA-Z]\s+', ' ', x))
    
    #create a final text field to work on
    data['final_text'] = data['processed_text_5']


# In[15]:


#pre-processing or cleaning data

text_preprocessing(df)

df.head()


# In[16]:


#create tokenized data for LDA

df['final_tokenized'] = list(map(nltk.word_tokenize, df.final_text))

df.head()
