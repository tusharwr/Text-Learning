#!/usr/bin/env python
# coding: utf-8

# # Background																			
# 
# - 1	The adjacent 3 tabs contain a data dump of search strings used by EXP clients to access relevant content available on Gartner.com for the months of August, September and October in the year 2018. Every row mentions if the EXP client is "Premium" or not, Persona (that was used for data extraction for EXP clients from main database), day on which the search string was used and finally the search string. In total there are 68544 rows of data available across all the months.																		
# ## Task 1																			
# 
# - 2	Clean the dataset using standard text cleaning steps and process the data to allow for the following analysis.																		
# - 3	Identify the most popular topics being searched for by EXP clients and plot the top 10 topics by their frequency of occurrence.																		
# 
# - 4	Report on the volume growth of these topics over August, September and October.																		
# ## Task 2																			
# 
# - 5	Used the cleaned dataset from Step 2 and process your dataset for the following analysis. 																		
# - 6	Use the concept of Named Entity Recognition in your code to identify a list of geographies and organizations being mentioned in the search terms.																		
# 
# - 7	Plot the geographies and organizations by their frequency of occurrence (count 1 mention of a geography, if the same geography is mentioned more than once in the same search string). If you can do it for the mention of "technologies" such as ai, analytics etc. then it will be a plus. Any useful trends observed in these mentions of geographies, organizations and technologies should be plotted and presented.																		
# 
# # Final Output & Next Steps																			
# 
# - 8	"Final output of the exercise should include
# 
#  *1. 3 Code files- 1 used for data cleaning and 2 used for each of the 2 tasks (with data processing and data analysis). Please comment your code appropriately. You will be evaluated for properly structuring your code and for building checks and balances in your analysis- which should be included in your code as well.*
# 
#  *2. If some data visualization tool such as Tableau/PowerBI is used for presentation of the plots in the panel round (if selected) then it will be considered a plus for the candidate. PPT presentation is acceptable though. The following visualizations are required- *
# 
# **- Please prepare 1-2 slides to explain your data cleaning and processing steps, 1-2 slides to display the results of Task 1 (include the methodology used for completing the task), 1-2 slides to display the result of Task 2 (include the methodology used for completing the task), 1-2 slides on what other analysis is possible on the data set including the expected insights from those (for this you will need to mention the preferred methodology for text analysis). "**																		
# - 9	You will be given a time limit of 3 Days from the time this test is given, to prepare the output. The candidates should upload the output docs- Dashboard/PPT & their 3 code files in a G-drive link and send them across to the assigned recruiter.																		
# - 10	If your output gets selected, you will be asked to present your findings & approach to our panel of experts who would cross question you on your analysis.																		
# 

# In[26]:


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


# ## LDA training 

# In[17]:


# Create Dictionary

id2word = corpora.Dictionary(df['final_tokenized'])

texts = df['final_tokenized']

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])


# In[18]:


id2word[0]


# In[19]:


# Human readable format of corpus (term-frequency)
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]


# In[20]:


get_ipython().run_cell_magic('time', '', "\nnum_topics = 10\n\nlda_model = LdaMulticore(corpus=corpus,\n                         id2word=id2word,\n                         num_topics=num_topics, \n                         workers=3,              #CPU cores\n                         random_state=100,\n                         chunksize=400,         #Number of documents to be used in each training chunk.\n                         passes=40,              #Number of passes through the corpus during training.\n                         alpha='asymmetric',\n                         per_word_topics=True)")


# In[27]:


# View the topics in LDA model

pp.pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


# #### What is topic coeherence
# 
# https://rare-technologies.com/what-is-topic-coherence/
# 
# What exactly is this topic coherence pipeline thing? Why is it even important? Moreover, what is the advantage of having this pipeline at all? In this post I will look to answer those questions in an as non-technical language as possible. This is meant for the general reader as much as a technical one so I will try to engage your imaginations more and your maths skills less.
# 
# Imagine that you get water from a lot of places. The way you test this water is by providing it to a lot of people and then taking their reviews. If most of the reviews are bad, you say the water is bad and vice-versa. So basically all your evaluations are based on reviews with ratings as bad or good. If someone asks you exactly how good (or bad) the water is, you blend in your personal opinion. But this doesn’t assign a particular number to the quality of water and thus is only a qualitative analysis. Hence it can’t be used to compare two different sources of water in a definitive manner.
# 
# Since you are a lazy person and strive to assign a quantity to the quality, you install four different pipes at the end of the water source and design a meter which tells you the exact quality of water by assigning a number to it. While doing this you receive help from a lot of wonderful people around you and therefore you are successful in installing it. Hence now you don’t need to go and gather hundred different people to get their opinion on the quality of water. You can get it straight from the meter and this value is always in accordance with the human opinions.
# 
# The water here is the topics from some topic modelling algorithm. Earlier, the topics coming out from these topic modelling algorithms used to be tested on their human interpretability by presenting them to humans and taking their input on them. This was not quantitative but only qualitative. The meter and the pipes combined (yes you guessed it right) is the topic coherence pipeline. The four pipes are:
# 
# Segmentation : Where the water is partitioned into several glasses assuming that the quality of water in each glass is different.
# Probability Estimation : Where the quantity of water in each glass is measured.
# Confirmation Measure : Where the quality of water (according to a certain metric) in each glass is measured and a number is assigned to each glass wrt it’s quantity.
# Aggregation : The meter where these quality numbers are combined in a certain way (say arithmetic mean) to come up with one number.
# And there you have your topic coherence pipeline! There are surely much better analogies than this one but I hope you got the gist of it.

# In[28]:


get_ipython().run_cell_magic('time', '', "\n# Compute Perplexity\nprint('\\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.\n\n# Compute Coherence Score\ncoherence_model_lda = CoherenceModel(model=lda_model, texts=df['final_tokenized'], dictionary=id2word, coherence='c_v')\ncoherence_lda = coherence_model_lda.get_coherence()\nprint('\\nCoherence Score: ', coherence_lda)")


# ## Top 10 topics by frequency of occurence
# 
# 

# In[29]:


get_ipython().run_cell_magic('time', '', '\n# Visualize the topics\n\npyLDAvis.enable_notebook()\nvis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)\nvis')


# #### How to find the optimal number of topics for LDA?  
# 
# My approach to finding the optimal number of topics is to build many LDA models with different values of number of topics (k) and pick the one that gives the highest coherence value.
# 
# Choosing a ‘k’ that marks the end of a rapid growth of topic coherence usually offers meaningful and interpretable topics. Picking an even higher value can sometimes provide more granular sub-topics.
# 
# If you see the same keywords being repeated in multiple topics, it’s probably a sign that the ‘k’ is too large.
# 
# The compute_coherence_values() (see below) trains multiple LDA models and provides the models and their corresponding coherence scores.

# If the coherence score seems to keep increasing, it may make better sense to pick the model that gave the highest CV before flattening out. This is exactly the case here.
# 
# So for further steps I will choose the model with 20 topics itself.

# ## Sub-Task2 Named Entity Recognition

# In[34]:


from IPython.display import Image
Image("img/picture.png")


# In[35]:


import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm

nlp = en_core_web_sm.load()


# In[36]:


#removing duplicates

final_text = df['final_text'].unique()

print('Number of Query Text: ', len(final_text))


# In[37]:


corpus = list(nlp.pipe(final_text))


# In[38]:


# Looking at number of times each ent appears in the total corpus
# nb. ents all appear as Spacy tokens, hence needing to cast as str

from collections import defaultdict

all_ents = defaultdict(int)

for i, doc in enumerate(corpus):
    #print(i,doc)
    for ent in doc.ents:
        all_ents[str(ent)] += 1
        #print(ent)
        
print('Number of distinct entities: ', len(all_ents))


# In[39]:


# labels = [x.label_ for x in corpus.ents]
# Counter(labels)

ent_label = []
ent_common = []

for i, doc in enumerate(corpus):
    for ent in doc.ents:
        ent_label.append(ent.label_)
        ent_common.append(ent.text)
        
print("Unique labels for entities : ", Counter(ent_label))
print("Top 3 frequent tokens     : ", Counter(ent_common).most_common(3))


# In[40]:


sentences = []

for i, doc in enumerate(corpus):
    for ent in doc.sents:
        sentences.append(ent)

print(sentences[0])


# In[41]:


# Most popular ents

import operator

sorted_ents = sorted(all_ents.items(), key=operator.itemgetter(1), reverse=True)
sorted_ents[:30]


# ### List of geographies and organizations being mentioned in the search terms.																									

# In[52]:


for i, doc in enumerate(corpus):
    for ent in doc.ents:
        if ent.label_ == 'ORG' or ent.label_ == 'GPE':
            print(ent.text, ent.start_char, ent.end_char, ent.label_)

