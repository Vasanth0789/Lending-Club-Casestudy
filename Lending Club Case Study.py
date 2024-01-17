#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from datetime import datetime
from matplotlib.pyplot import pie, axis, show
from matplotlib import pyplot as plt
import seaborn as sns
import scipy as scp
import warnings
warnings.filterwarnings('ignore')


# In[2]:


Loan_raw_data= pd.read_csv("loan.csv")


# # Filtering the unused columns in the dataset

# In[3]:


# Filtering the unused columns in the dataset

Loan_data_filtered = Loan_raw_data.filter(items=['member_id','loan_amnt','funded_amnt','funded_amnt_inv','term',
                                                 'int_rate','installment','grade','sub_grade',
                                    'emp_title','emp_length','home_ownership','annual_inc','verification_status',
                                    'issue_d','loan_status','pymnt_plan','url','desc','purpose','title','zip_code',
                                    'addr_state','dti','delinq_2yrs','earliest_cr_line','inq_last_6mths',
                                    'mths_since_last_delinq','mths_since_last_record','open_acc','pub_rec','revol_bal',
                                    'revol_util','total_acc','initial_list_status','out_prncp','out_prncp_inv',
                                    'total_pymnt','total_pymnt_inv','total_rec_prncp','total_rec_int','total_rec_late_fee',
                                    'recoveries','collection_recovery_fee','last_pymnt_d','last_pymnt_amnt',
                                    'next_pymnt_d','last_credit_pull_d','collections_12_mths_ex_med',
                                    'mths_since_last_major_derog','policy_code','application_type'
])


# In[4]:


Loan_data_filtered


# # Finding Columns with Null values

# In[5]:


Loan_data_filtered.columns[Loan_data_filtered.isnull().any()]


# In[6]:


Loan_data_cleaned=Loan_data_filtered.fillna("Undefined ")


# In[7]:


# checking the null/NAN data in cleaned dataset
Loan_data_cleaned.columns[Loan_data_cleaned.isnull().any()]


# In[8]:


# Filtering to Defaulters and Current borrowers for analysis

Loan_data =Loan_data_cleaned.query("loan_status== ['Current','Charged Off']")
Loan_data


# In[9]:


# Data Type Conversions of fields in data
Loan_data['annual_inc'] = Loan_data['annual_inc'].astype(float)
Loan_data["int_rate"]= Loan_data["int_rate"].str.rstrip('%')
Loan_data["int_rate"] = Loan_data["int_rate"].astype(float)
Loan_data["funded_amnt"] = Loan_data["funded_amnt"].astype(float)
Loan_data["funded_amnt_inv"] = Loan_data["funded_amnt_inv"].astype(float)
Loan_data["loan_amnt"] = Loan_data["loan_amnt"].astype(float)
Loan_data['issue_d']= Loan_data['issue_d']+"-"+"24"
Loan_data['issue_d'] = Loan_data['issue_d'].apply(pd.to_datetime).dt.date
Loan_data['MYissue_d']= Loan_data['MYissue_d']= pd.DatetimeIndex(Loan_data['issue_d']).year.astype(str)+'-'+ pd.DatetimeIndex(Loan_data['issue_d']).month.astype(str)


# In[10]:


# Univariate analysis on  Loan Status to see % of Charged off out of total data( which includes Current ongoing loans )

Loan_data.groupby('loan_status').size().plot(kind='pie', autopct='%.2f')


# In[11]:


plt.figure(figsize=(8,8))

sns.barplot(data =Loan_data,x='loan_amnt', y='home_ownership', hue ='loan_status',palette="pastel")
plt.show()


# In[12]:


# Filtering  Loan data to Charged off, which filters to defaulting Loan data to correlation between the variables
Loan_data_chargedoff = Loan_data[Loan_data['loan_status'] .str.contains("Charged Off")]
Loan_data_chargedoff


# In[13]:


# Univariate analysis:
Loan_data_chargedoff.groupby('term').size().plot(kind='pie', autopct='%.2f')


# In[ ]:


# Plotting a Correlation Map across all the fields in the datasource to see the correlation between them 
sns.set(style="ticks", color_codes=True)    
g = sns.pairplot(Loan_data_chargedoff)
plt.show()


# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(Loan_data.corr(),annot=True)
plt.show()


# In[ ]:


#Plotting Scatter Plot between Annual Income vs  Funded Amount to show the correlation for Loan Defaulting
import seaborn as sns
from matplotlib import pyplot as plt
sns.scatterplot(x="funded_amnt", y="annual_inc",palette="pastel", data=Loan_data_chargedoff)
plt.show()


# In[ ]:


#Plotting Scatter Plot between Total Payment vs  Funded Amount to show the correlation for Loan Defaulting
plt.figure(figsize=(10,10))
sns.scatterplot(x="funded_amnt", y="total_pymnt", palette="pastel",data=Loan_data_chargedoff)
plt.show()


# In[ ]:


# Creating a grouped dataset on Grade for analysis
Loan_data_Gradegrp = Loan_data_chargedoff.groupby(["grade"])["int_rate"].max()
Loan_data_Gradegrp = pd.DataFrame(Loan_data_Gradegrp)
Loan_data_Gradegrp


# In[ ]:


# Plotting an insight to show Interest variation across different grades
sns.barplot(data =Loan_data,x='grade', y='int_rate',palette="pastel")
plt.title("% Interest by Grade")
plt.xlabel("Grade")
plt.ylabel("Interest rate")
plt.show()


# In[ ]:


# Converting Funded Amount to Float datatype for aggregation and creating a grouped dataset on Grade for analysis
Loan_data_Emplenggrp = Loan_data_chargedoff.groupby(["emp_length"])["funded_amnt"].sum()
Loan_data_Emplenggrp = pd.DataFrame(Loan_data_Emplenggrp)
Loan_data_Emplenggrp


# In[ ]:


#Loan_data_Emplenggrp['funded_amnt'].plot(kind="bar")
plt.figure(figsize=(10,10))
sns.barplot(data =Loan_data_Emplenggrp,x='emp_length', y='funded_amnt',palette="pastel")
plt.title("Funded Amount by Employee Tenure")
plt.xlabel("Employee Length")
plt.ylabel("Funded Amount")
plt.show()


# In[ ]:


# Multi Variate  Analysis
# Grouping for Funded Amount,Loan Amount, Funded Amount Invested and Total payment amount by Issue Date
Loan_data_measgrp= Loan_data_chargedoff.groupby("MYissue_d")[["funded_amnt","loan_amnt","funded_amnt_inv","total_pymnt"]].sum()
Loan_data_measgrp


# In[ ]:


Loan_data_measgrp.plot.line();


# In[ ]:


# Distribution of Charged Off loans across States
plt.figure(figsize=(25,10))
sns.countplot(x='addr_state', data=Loan_data_chargedoff,hue= 'verification_status')


# # DTI: Debt to Income Ratio

# In[ ]:


plt.figure(figsize=(10,10))
sns.barplot(data =Loan_data_chargedoff,x='MYissue_d', y='dti',palette="pastel")
plt.title("DTI")
plt.xlabel("Month Year")
plt.ylabel("DTI")
plt.show()

