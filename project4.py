#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd


# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


df = pd.read_csv('Cleaned_DS_Jobs.csv')
df.head()


# In[6]:


df.columns


# In[7]:


df['job_simp'].value_counts()


# In[8]:


df.info()


# In[9]:


df.job_state.value_counts()


# In[10]:


df.sort_values(by='min_salary')


# In[11]:


df.describe()


# In[ ]:


sns.pairplot(df)


# In[ ]:


df.min_salary.hist()


# In[ ]:


sns.boxplot(x=df["avg_salary"])


# In[ ]:


df = df[(df.avg_salary>50)&(df.avg_salary<200)]
sns.boxplot(x=df["avg_salary"])


# In[ ]:


df.shape


# In[ ]:


sns.lmplot(x='company_age',y='avg_salary',data=df)


# In[ ]:


import numpy as np
fig, ax = plt.subplots(figsize=(10,10)) 
matrix = np.triu(df.corr())
sns.heatmap(df.corr(),annot=True, fmt='.1g', vmin=-1, vmax=1, center= 0,  ax=ax)


# In[ ]:


fig, ax = plt.subplots(figsize=(10,10)) 
matrix = np.triu(df.corr())
sns.heatmap(df.corr(),annot=True, fmt='.1g', vmin=-1, vmax=1, center= 0, mask=matrix, ax=ax)


# In[ ]:


fig, ax = plt.subplots(figsize=(10,10)) 
chart = sns.barplot(x=df.Industry.value_counts().index, y=df.Industry.value_counts())
_=chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
plt.savefig('job industry count.png')


# In[ ]:


fig, ax = plt.subplots(figsize=(10,10)) 
chart = sns.barplot(x=df.Sector.value_counts().index, y=df.Sector.value_counts())
_=chart.set_xticklabels(chart.get_xticklabels(), rotation=75)

plt.savefig('job sector count.png')


# In[ ]:


fig, ax = plt.subplots(figsize=(10,10)) 
chart = sns.barplot(x=df.job_state.value_counts().index, y=df.job_state.value_counts())
_=chart.set_xticklabels(chart.get_xticklabels(), rotation=45)

plt.savefig('job state count.png')


# In[ ]:


fig, ax = plt.subplots(figsize=(10,10)) 
chart = sns.barplot(x=df.job_simp.value_counts().index, y=df.job_simp.value_counts())
_=chart.set_xticklabels(chart.get_xticklabels(), rotation=35)

plt.savefig('job title count.png')


# In[ ]:


fig, ax = plt.subplots(figsize=(10,10)) 
chart = sns.barplot(x=df.seniority.value_counts().index, y=df.seniority.value_counts())
_=chart.set_xticklabels(chart.get_xticklabels(), rotation=90)

plt.savefig('sr.png')


# In[ ]:


pd.pivot_table(df, index='job_simp', values='avg_salary')


# In[ ]:


pd.pivot_table(df, index=['job_simp','seniority'], values='avg_salary').sort_values('avg_salary', ascending =False)


# In[ ]:


pd.pivot_table(df, index='job_state', values='avg_salary').sort_values('avg_salary', ascending =False)


# In[ ]:


pd.set_option('display.max_rows', None)
pd.pivot_table(df, index=['job_state', 'job_simp'], values='avg_salary', aggfunc='count').sort_values('job_state', ascending =False)


# In[ ]:


pd.pivot_table(df[df.job_simp=='data scientist'], index='job_state', values='avg_salary').sort_values('avg_salary', ascending =False)


# In[ ]:


df.head()


# In[ ]:


from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk


# In[ ]:


words = " ".join(df['Job Description'])
wordcloud = WordCloud(max_words=5000, width =1280, height = 720, background_color="black").generate(words)
plt.figure(figsize=[15,15])
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
plt.savefig('wordcloud.png')


# In[ ]:


sns.lmplot(x='Rating',y='avg_salary',data=df)


# In[ ]:


df.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




