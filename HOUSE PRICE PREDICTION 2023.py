#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import skew,norm
from scipy.stats.stats import pearsonr


# In[2]:


df=pd.read_csv('kc_house_data.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# # Univariate Analysis

# In[5]:


df.hist(bins=15,color="green", edgecolor="blue", linewidth =1,xlabelsize = 8, ylabelsize = 8, grid = False)
plt.tight_layout(rect=(0,0,1.2,1.2))
plt.suptitle("house price prediction univariate plot", x= 0.65, y= 1.25,fontsize= 12)


# # multi variate analysis

# In[6]:


corr= df.corr


# In[7]:


corr()


# In[8]:


# Assuming 'df' is your DataFrame and you want to calculate the correlation matrix
corr = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm_r", linewidths=0.2)
plt.title("house price multivariate plot", fontsize=12)
plt.show()


# # handling outliers

# In[9]:


df.hist(bins=15,color="green", edgecolor="blue", linewidth =1,xlabelsize = 8, ylabelsize = 8, grid = False)
plt.tight_layout(rect=(0,0,1.2,1.2))
plt.suptitle("house price prediction univariate plot", x= 0.65, y= 1.25,fontsize= 12)


# In[10]:


plt.scatter(df["bathrooms"], df["price"], color="green")


# # Observation
#   out liers in:
#   
#   . id
#   . zipcode
#   . bathrooms

# In[11]:


def p_outliers (df,col,uv_f=3,lv_f=0.3):
    uv=np.percentile(df[col],[99])*uv_f
    lv=np.percentile(df[col],[99])*lv_f
    return f"upper_limit: {uv} ----- lower_limit: {lv}"


# In[12]:


#zipcode
p_outliers(df,["bathrooms"])


# In[13]:


df["bathrooms"][(df.bathrooms) > 1.275]= df.bathrooms.mean()


# In[14]:


df["bathrooms"][(df.bathrooms) > 1.275]


# In[15]:


#zipcode
p_outliers(df ,"zipcode")


# In[16]:


df["zipcode"][(df.zipcode<29459)]=df.zipcode.mean()


# In[17]:


df["zipcode"][(df.zipcode<29459)]


# # Filling missing column

# In[18]:


df.info()


# In[22]:


df.lat=df.lat.fillna (df.lat.mean())


# In[21]:


df["avg_sqft"]=(df.sqft_lot+df.sqft_above+df.sqft_basement+df.sqft_living15+df.sqft_lot15)/5


# # Handling categorical data

# In[25]:


df.zipcode.unique()


# In[26]:


del df["zipcode"]


# In[27]:


df=pd.get_dummies(df,columns=["bedrooms","bathrooms"])


# In[29]:


df.head()


# In[30]:


del df["waterfront"]


# In[31]:


del df["view"]


# In[32]:


df.head()

