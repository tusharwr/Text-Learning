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
