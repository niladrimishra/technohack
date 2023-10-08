#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


df=pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')


# In[8]:


df.head()


# In[9]:


df.shape
df.info()


# In[11]:


df[df["TotalCharges"].str.contains(" ")]


# In[13]:


df.info()


# In[14]:


# exclude rows with TotalCharges culomns contains white space
df = df.loc[~df['TotalCharges'].str.contains(' ')]

# transform TotalCharge col to float
df['TotalCharges'] = df['TotalCharges'].astype(float)


# In[15]:


df.isnull().sum()


# In[16]:


df.duplicated().sum()


# # Observations:
# 
# The raw data contains 21 columns with 7,043 rows.
# 
# After checking, there are no duplicated data; however,it was found that 11 columns of TotalCharges contain white spaces and need to be removed. Consequently, there are 7,032 rows remaining.
# 
# The TotalCharges column has an inappropriate data type and needs to be converted to float.
# 
# Except for the columns SeniorCitizen, Tenure, MonthlyCharges, and TotalCharges, all other columns are categorical columns.

# stat

# In[17]:


numerical_features = []
categorical_features = []

for i in df.columns:
    if (df[i].dtype == 'object'):
        categorical_features.append(i)
    else:
        numerical_features.append(i)


# In[18]:


numerical_summary = df[numerical_features].describe()
numerical_summary


# #Observation:
# 
# Overall, the minimum and maximum values make sense for each column
# 
# SeniorCitizen column is boolean/binary column since the value is 0 or 1, no need to conclude its simmetricity.
# 
# Tenure, MonthlyCharges, and TotalCharges are discrete with continue values
# 
# Tenure and TotalCharges are positively skewed distribution
# 
# MonthlyCharges has negatively skewed distribution

# In[20]:


categorical_summary = df[categorical_features].describe()
categorical_summary


# In[21]:


# showing the precise value counts
# this code is especially useful if we have many categorical columns
for col in categorical_features:
  print(f"Value counts of {col} column")
  print(df[col].value_counts(), '\n')


# # Observastions:
# 
# Partner, Dependents, PhoneService, and Churn have 2 unique values: 'yes' and 'no', whereas gender has 2 unique values: 'Male' and 'Female'.
# 
# MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, Techsupport, StreamingTV, StreamingMovies, and Contract have 3 unique values.
# 
# PaymentMethod has 4 unique values: Electronic check, Mailed check, Bank transfer (automatic), and Credit card (automatic).
# 
# A total of 1520 customers do not have internet service.
# 
# The majority of customers choose a Contract of month-to-month with PaperlessBilling using the PaymentMethod of electronic check.

# Univariate Analysis
# Boxplot to Detect Outliers

# In[38]:


# adjust the figure size for better readability
plt.figure(figsize=(12,6))

# plotting
features = numerical_features
for i in range(0, len(features)):
    plt.subplot(1, len(features), i+1)
    sns.boxplot(y=df[features[i]], color='purple')
    plt.tight_layout()


# In[23]:


# check the outlier full profile
outlier = df['SeniorCitizen'].max()

df[df['SeniorCitizen'] == outlier]


# # Observation:
# 
# No outlier data was found in the Tenure, MonthlyCharges, and TotalCharges columns
# There are a total of 1142 outliers in the SeniorCitizen column due to its data type being boolean, which only contains values 0 or 1. This can be left as it is

# KDE Plot for Knowing The Distribution Form

# In[26]:


# KDE lebih untuk feature yang sifatnya continue (in this case gre_score, toefl_score, gpa)
# adjust the figure size for better readability
plt.figure(figsize=(16,8))

features = numerical_features
for i in range(0, len(features)):
    plt.subplot(2, len(features)//2 + 1, i+1)
    sns.kdeplot(x = df[features[i]], color = 'green')
    plt.xlabel(features[i])
    plt.tight_layout()


# # Observation:
# 
# We can ignore interpreting feature columns with limited discrete values such as SeniorCitizen
# 
# most customers are (distribution peak):
# 
# Senior citizen
# 5 months tenure
# 20 dollars monthly charges
# 100 dollars total charges

# Bivariate and Multivariate Analysis

# In[27]:


def create_histograms(df, features):
    for feature in features:
        plt.figure(figsize=(5, 3))
        sns.histplot(data = df, x = feature, hue = 'Churn', multiple = 'dodge', bins=30)
        plt.title(f'Histogram of {feature} with Churn')
        plt.xlabel(feature)
        plt.ylabel('Frequency')

features = ["gender","Partner","Dependents","SeniorCitizen"]
create_histograms(df, features)


# In[28]:


features = ["PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
create_histograms(df, features)


# In[29]:


features = ["Contract","PaperlessBilling","PaymentMethod"]
create_histograms(df, features)


# In[37]:


# correlation heatmap
numeric_df = df.select_dtypes(include=['number'])  # Memilih hanya kolom numerik
correlation = numeric_df.corr()
plt.figure(figsize=(8, 6))
plt.title("Correlation Heatmap")
sns.heatmap(correlation, annot=True, fmt=".2f", cmap="Blues")
plt.show()


# # Observation:
# 
# It can be concluded that there is a relatively high correlation between tenure and TotalCharges, as well as between MonthlyCharges and TotalCharges

# In[34]:


# pairplot of the data
sns.pairplot(df, hue = 'Churn')


# # Observation:
# 
# data tidak memberikan insight yang begitu berarti, karena tidak terseperasi dengan baik
# 
# senior citizen memutuskan untuk berhenti berlangganan dengan tenure yang singkat
# 
# customer cenderung berhenti berlangganan di awal tenure dengan monthly charges yang tinggi

# Deep Dive Exploration
# 1. Does the internet service provider affect tenure and monthly charges?

# In[35]:


tenure_by_internet = df.groupby('InternetService').agg(
    mean_tenure = ('tenure','mean'),
    mean_monthlycharges = ('MonthlyCharges', 'mean')).reset_index()
tenure_by_internet


# # It can be observed that users with a fiber optic internet service provider subscribe for a longer period compared to DSL and those without an internet service. This could be due to the better quality, stability, and speed of fiber optic internet compared to others.
# 
# 2. Can the partners and dependents of customers affect their tenure and total charges?

# In[36]:


agg_data = df.groupby(['Partner','Dependents']).agg(
    mean_tenure = ('tenure', 'mean'),
    mean_totalcharges = ('TotalCharges', 'mean')
).reset_index()
agg_data


# Based on the table above, it is known that the highest values of mean_totalcharges and mean_tenure are found in customers who have partners but do not have dependents.

# # EDA CONCLUSION

# The raw dataset used contains missing values in the TotalCharge column, which contains white spaces, and its data type should be float but is identified as an object.
# There are no duplicate data entries.
# Gender does not have a significant impact on churn; both females and males have similar proportions of churn and non-churn customers.
# In the partner column, the highest churn rate is observed among customers who do not have a partner, while customers with partners tend to have lower churn rates.
# The presence or absence of dependents influences whether a customer churns or not. Customers without dependents are less likely to churn.
# Customers without internet services have lower monthly charges and shorter tenures compared to those with DSL and fiber optic services.
# The more services a customer subscribes to, the higher their total charges tend to be.
