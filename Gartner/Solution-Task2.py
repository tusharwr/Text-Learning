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

