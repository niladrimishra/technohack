#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df=pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.head()


# In[4]:


df.describe()


# In[5]:


df.shape


# In[6]:


df.isnull().sum()


# In[7]:


df.head()


# In[8]:


df.info()


# In[9]:


##checking if we have any duplicate records 
df['EmployeeNumber'].duplicated().sum()


# In[10]:


## glancing unique values for each column in the data set 
for column in df.columns:
    unique_values = df[column].unique()
    print(f"Unique values in {column}: {unique_values}")


# # General observations about dataset
# After examining the values present in each column, we have the following observations:
# 
# -Data is provided for only three departments: 'Sales,' 'Research & Development,' and 'Human Resources.'
# 
# -Some variables are coded with numbers, but they represent ordinal data. For example, education level is coded as 1, 2, 3, 4, or 5.
# 
# -'Employee Count' and 'Employee Number' are identifiers, so we can remove them.
# 
# -'Over18' also has only one value, 'Y,' so we can exclude it from the analysis.
# 
# -'Standard Hours' has only one value, which is 80, so we can safely remove this variable.
# 
# -Regarding 'Performance Rating,' it appears to have only values 3 and 4. Typically, we would expect values in the range of 1 to 5. We may want to check for missing records in this variable. However, for our current analysis, we will assume that the data is complete.
# 
# In summary, the dataset appears to be clean with no missing values or duplicates. Let's begin by removing the unnecessary variables.

# In[11]:


## dropping columns 
cols_to_drop = ['EmployeeCount','EmployeeCount','Over18','StandardHours']
df = df.drop(columns = cols_to_drop )


# # We need to find the drivers of attrition. So first let us find the overall attrition percentage

# In[12]:


print(df['Attrition'].unique())
df['Attrition']= df['Attrition'].replace(['Yes' ,'No'],[1,0]) ##converting to numeric


# In[14]:


round(df['Attrition'].mean()*100,1)


# # The overall attrition percentage is 16.1%.
# 
# The attrition percentage varies from industry to industry, and we don't have any historical trends to compare this year's attrition rate to. After conducting some research online, the ideal attrition rate is considered to be between 10% and 12%. Given this benchmark, attrition does appear to be somewhat high. Therefore, let's analyze the data and attempt to identify patterns that may indicate high attrition in specific segments. This analysis will enable management to take actionable steps to control attrition.
# 
# From a company perpective , let us first explore where is a attrition coming from ?
# 
# Department , Level , Job Role
# years at company
# Gender
# Job Satisfaction
# Compensation

# In[15]:


df.columns


# In[16]:


#which department has highest attrition
df.groupby('Department')['Attrition'].agg(['sum','count','mean']).sort_values(by='mean', ascending = False).reset_index()


# # Observations
# 
# Attrition in the Sales department is 20%, despite having half the number of employees compared to the R&D department. This suggests there may be an issue that requires further investigation. Attrition is also high in HR, but due to the smaller number of employees in this department, the percentage is high primarily because of the lower denominator. The R&D department has the majority of employees, and attrition is approximately 13.4%, slightly higher than the benchmark of 10-12%

# In[17]:


##Level analysis vs.attrition
df.groupby('JobLevel')['Attrition'].agg({'sum','count','mean'}).reset_index()


# # Observations
# 
# Attrition is high at 26% for employees at job level 1. This high attrition rate translates into increased recruitment and onboarding costs. If we have exit surveys for level 1 employees, valuable insights can be derived to understand what is triggering attrition at this junior entry level—whether it's a mismatch between the job they were hired for, incorrect expectations, or other factors. Additionally, employees at job level 3 are leaving at a faster rate, approximately 14.6%

# In[18]:


##Job Role vs. attrition
df.groupby('JobRole')['Attrition'].agg({'sum','count','mean'}).sort_values(by= 'mean', ascending = False).reset_index()


# In[19]:


## deep dive with department and job role
df.groupby(['Department','JobRole'])['Attrition'].agg({'sum','count','mean'}).reset_index()


# In[20]:


## deep dive with department, job role and job level
df.groupby(['Department','JobRole','JobLevel'])['Attrition'].agg({'sum','count','mean'}).reset_index()


# # Observations
# 
# While analyzing, we examine both percentages and sums/counts. In the HR department, the attrition rate is 30% for level 1 employees. In the R&D department, specifically for Research Scientists at level 1, there is an attrition rate of 19%, and for Lab Technicians at level 1, the attrition rate is 28%. In the Sales department, Sales Representatives at level 1 have an attrition rate of 42%, while Sales Executives at level 4 have an attrition rate of 28%. Executives leaving at senior levels can incur significant costs for the company. Furthermore, executives across all levels have an above-average attrition rate.
# 
# It is evident that attrition is mainly concentrated among junior-level employees, particularly at level 1, and within the Sales department, especially in the Sales Representative role, followed by Sales Executives. Additionally, level 1 Research Scientists experience a 20% attrition rate, while Lab Technicians also have a high attrition rate.

# In[21]:


#Business Travle vs Attrition
df.groupby(['BusinessTravel'])['Attrition'].agg({'sum','count','mean'}).reset_index()


# # Attrition percentage is high (24%) among employees who have to travel frequently. We assume that the sales department would involve a lot of travel. Let's delve deeper to validate this assumption.

# In[22]:


##adding other variables to business travel
df.groupby(['Department','BusinessTravel'])['Attrition'].agg({'sum','count','mean'}).reset_index()


# # Observations
# 
# It's observed that across all departments, roles that require frequent travel, as well as roles that involve rare travel, experience high levels of attrition. This raises questions about whether employees are adequately compensated for travel allowances or if the travel arrangements themselves are not conducive to employee satisfaction. Given that a vast majority of employees are subjected to business travel, it is recommended to re-evaluate the travel policy.

# In[24]:


### Tenure or time duration also plays a very important role. It is usually seen that younger employees have higher attrition 
tenure_cols= ['Attrition','TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole']
Tenure_df= df[tenure_cols]

##let us do some binning 0-1 yrs,1-2 yrs,,2-5 yrs,5-10 yrs ,10-15 yrs

bins = [0, 2, 5, 10,50]

# Define bin labels
bin_labels = [ '0-2', '2-5', '5-10','10-50']

# Use cut() function to bin the data
Tenure_df['TotalWorkingYears_Bin'] = pd.cut(Tenure_df['TotalWorkingYears'], bins=bins, labels=bin_labels, right=False)
Tenure_df['YearsAtCompany_Bin'] = pd.cut(Tenure_df['YearsAtCompany'], bins=bins, labels=bin_labels, right=False)
Tenure_df['YearsInCurrentRole_Bin'] = pd.cut(Tenure_df['YearsInCurrentRole'], bins=bins, labels=bin_labels, right=False)


# In[25]:


Tenure_df.groupby('TotalWorkingYears_Bin')['Attrition'].agg({'sum','count','mean'})


# In[26]:


Tenure_df.groupby('YearsAtCompany_Bin')['Attrition'].agg({'sum','count','mean'})


# In[28]:


Tenure_df.groupby('YearsInCurrentRole_Bin')['Attrition'].agg({'sum','count','mean'})


# Usually, 'total working years' and 'years at the company' should mean the same thing. Let's focus on 'years at the company' and 'years in the current role' for analys

# In[29]:


## Also let's look whether the company has a higher percentage of young or senior individuals
Tenure_df['YearsAtCompany_Bin'].value_counts(normalize = True)


# Observations -
# 
# 14% of the workforce has a tenure of 0-2 years, but attrition in this group is 34%.
# 
# 18% of the workforce has a tenure of 2-5 years, and attrition in this group is 16%.
# 
# 60% of the workforce has a tenure of more than 5 years, with attrition averaging around 10-11%.
# 
# Therefore, attrition is primarily concentrated among employees with a tenure of 0-2 years. Attrition reduces to half in the 2-5 year group, and after employees spend more than 5 years with the company, attrition stabilizes at an acceptable level of 10%

# In[30]:


#compensation_variables 
#satisfaction variables

df.columns
compensation_variables =['Attrition','HourlyRate','MonthlyIncome', 'MonthlyRate', 'StockOptionLevel']
rewards = ['Attrition', 'PercentSalaryHike', 'StockOptionLevel','YearsSinceLastPromotion']
satisfaction_variables =['Attrition','EnvironmentSatisfaction','JobInvolvement','JobSatisfaction','RelationshipSatisfaction','WorkLifeBalance']


# In[32]:


##let us look ar satisfaction variables first 
sat_score= df[satisfaction_variables].groupby('Attrition').mean()
sat_score


# Observation - Overall, the mean satisfaction scores are lower for those who have left compared to those who are still working. This is not surprising, but more importantly, it signifies that people who feel less satisfied are more likely to leave the organization. Hence, the organization should continually monitor satisfaction scores through pulse surveys or other surveys

# In[41]:


#let us look at compensation variables 

fig, axes = plt.subplots(nrows=2, ncols=2, constrained_layout=True)
#subplot1
sns.boxplot(x= 'Attrition',y ='DailyRate' , data= df , palette = 'Set1',ax= axes[0,0])
axes[0,0].set_title('Box Plot for DailyRate  vs. Attrition')
#subplot2
sns.boxplot(x= 'Attrition',y ='HourlyRate' , data= df , palette = 'Set1',ax= axes[0,1])
axes[0,1].set_title('Box Plot for HourlyRate vs. Attrition')
#subplot3
sns.boxplot(x= 'Attrition',y ='MonthlyIncome' , data= df , palette = 'Set1',ax= axes[1,0])
axes[1,0].set_title('Box Plot for MonthlyIncome vs. Attrition')
#subplot4
sns.boxplot(x= 'Attrition',y ='MonthlyRate' , data= df , palette = 'Set1',ax= axes[1,1])
axes[1,1].set_title('Box Plot for MonthlyRate vs. Attrition')


# Observations - Hourly and monthly rates are almost the same. However, for monthly income, we see that people who left had significantly lower median compensation. Intuitively, it appears that employees who are leaving may have low compensation as one of the reasons for their departure. As employees who left were mostly junior this is not a surprise.

# In[42]:


rewards = ['Attrition', 'PercentSalaryHike', 'StockOptionLevel','YearsSinceLastPromotion']


# In[43]:


##plotting reards and other missed variables which could not be clubbed 

fig, axes = plt.subplots(nrows=3, ncols=2, constrained_layout=True,figsize=(8, 8))
sns.boxplot(x= 'Attrition',y ='PercentSalaryHike' , data= df ,palette = 'Set1',ax= axes[0,0])
axes[0,0].set_title('Box Plot for PercentSalaryHike  vs. Attrition')
sns.histplot(x ='StockOptionLevel' , hue= 'Attrition',multiple = 'stack',data= df ,palette = 'Set1',ax= axes[0,1])
axes[0,1].set_title('Box Plot for StockOptionLevel  vs. Attrition')
#subplot3
sns.boxplot(x= 'Attrition',y ='YearsSinceLastPromotion' , data= df , palette = 'Set1',ax= axes[1,0])
axes[1,0].set_title('Box Plot for YearsSinceLastPromotion vs. Attrition')
#subplot4
sns.boxplot(x= 'Attrition',y ='Age' , data= df, palette = 'Set1',ax= axes[1,1])
axes[1,1].set_title('Box Plot for Age vs. Attrition')
#subplot5
sns.boxplot(x= 'Attrition',y ='DistanceFromHome' , data= df , palette = 'Set1',ax= axes[2,0])
axes[2,0].set_title('Box Plot for DistanceFromHome vs. Attrition')
#subplot6
sns.countplot(x='OverTime' , hue= 'Attrition',data= df , palette = 'Set1',ax= axes[2,1])
axes[2,1].set_title('Box Plot for OverTime vs. Attrition')


# observations It is surprising that the people who left have the same average salary hike and were also likely promoted at the same speed as the people who did not leave. However, the people who left had little to no stock options. We also notice that people who left had longer commutes, which seems to be an issue when combined with those who have to travel frequently for business.

# # Predicting Employee Attrition
# Up to this point, we have conducted univariate and bivariate analyses and have identified some of the areas where the organization needs to focus to control attrition. We can utilize machine learning to create a predictive model that will help us determine which employees are likely to leave. This is a classification problem since the outcome variable, 'attrition,' has two possible outcomes: 'Yes' or 'No.' While attrition is a human decision influenced by personal reasons, it is also affected by changing macroeconomic conditions. Therefore, a model can provide an abstraction but not a 100% prediction. We are aware that two years ago, companies were in a hiring frenzy, but in 2023, the situation reversed, and along with the recession, the job market has become challenging. Attrition trends are continuously evolving. With that being said, let's start by examining the correlation plot, removing highly correlated variables, and preparing the data for modeling.

# In[46]:


# for machine learning , all datatypes should be numeric so we apply one hot encoding 

df = pd.get_dummies(df, drop_first =  True)


# In[47]:


#Heatmap

#calculate the correlation matrix
correlation_matrix= df.corr()

#set up the heatmap figure 
plt.figure(figsize = (24,16))
heatmap_= sns.heatmap(correlation_matrix, annot = True, cmap= 'coolwarm', fmt= '.1f')
heatmap_


# In[48]:


threshold = 0.8  # Adjust the threshold as needed
# Create a correlation mask
correlation_mask = (correlation_matrix.abs() > threshold) & (correlation_matrix != 1.0)

# Extract highly correlated variables
highly_correlated_vars = []

for col in correlation_matrix.columns:
    correlated_cols = correlation_mask.index[correlation_mask[col]]
    if len(correlated_cols) > 0:
        highly_correlated_vars.append(col)
        highly_correlated_vars.extend(correlated_cols)

highly_correlated_vars = list(set(highly_correlated_vars))

print("Highly correlated variables:", highly_correlated_vars)


# ##The two highly correlated variables are Monthly income and Job level as per data. We might want to keep both
# ## both variables are serving different puposes 

# In[50]:


# Splitting the data 

# Separate the input features (X) and the target variable (y)
X = df.drop(columns=['Attrition']) # X contains all columns except 'Attrition_Yes'
y = df['Attrition'] # y contains only the 'Attrition_Yes' column, which is the target variable


# In[52]:


X


# In[53]:


y


# In[54]:


# Splitting into testing and training sets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)


# In[55]:


# Checking the shapes:

print('X_train.shape',X_train.shape)
print('y_train.shape',y_train.shape)
print('X_test.shape',X_test.shape)
print('y_test.shape',y_test.shape)


# In[56]:


#Logistic Regression 


# In[57]:


# Importing the logistic regression classifier
from sklearn.linear_model import LogisticRegression

# Creating an instance of the logistic regression classifier
# Setting the random_state to ensure reproducibility of results
classifier_lr = LogisticRegression(random_state=0)


# In[58]:


# Training the model

classifier_lr.fit(X_train, y_train)


# In[59]:


# Making the prediction

y_pred = classifier_lr.predict(X_test)


# In[60]:


# Loading the metrics
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

# Calculate accuracy using the predicted and true target values (y_pred and y_test)
acc = accuracy_score(y_test, y_pred)

# Calculate F1 score using the predicted and true target values (y_pred and y_test)
f1 = f1_score(y_test, y_pred)

# Calculate precision score using the predicted and true target values (y_pred and y_test)
prec = precision_score(y_test, y_pred)

# Calculate recall score using the predicted and true target values (y_pred and y_test)
rec = recall_score(y_test, y_pred)


# In[61]:


# Getting the results

results = pd.DataFrame([['LogisticRegression', acc, f1, prec, rec]],
                       columns = ["Model", "accuracy", "f1", "precision", "recall"])

results


# In[62]:


# Checking the confusion matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[63]:


# Cross Validation

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier_lr, X=X_train, y=y_train, cv=10)
print("Accuracy is {:.2f}%".format(accuracies.mean()*100))
print("Standard Deviation is {:.2f}%".format(accuracies.std()*100))


# Summary
# 
# Sales department has a 20% attrition rate despite having fewer employees than R&D, suggesting a potential issue that requires investigation.
# 
# HR department has a 30% attrition rate among level 1 employees, mainly due to a smaller denominator.
# 
# R&D department has the majority of employees with a 13.4% attrition rate, slightly higher than the benchmark.
# 
# Attrition is high at 26% for job level 1 employees, indicating potential recruitment and onboarding cost challenges.
# 
# Job level 3 employees are leaving at a rate of approximately 14.6%.
# 
# Junior-level employees, especially at level 1, experience high attrition in Sales and R&D departments.
# 
# Frequent travelers exhibit a 24% attrition rate, suggesting the need to investigate travel-related factors.
# 
# Roles involving both frequent and rare travel across departments show high attrition, pointing to potential compensation and travel policy concerns.
# 
# Attrition is highest among employees with 0-2 years of tenure (34%), decreasing to 16% for 2-5 years and stabilizing at 10-11% for more than 5 years.
# 
# Despite similar salary hike and promotion rates, employees who left had lower stock options and longer commutes, possibly influencing attrition.
# 
# Thank you for reading. Please upvote if you liked it or leave a critique for this report so I can improve

# In[ ]:




