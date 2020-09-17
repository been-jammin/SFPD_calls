#!/usr/bin/env python
# coding: utf-8

# first, let's import all packages needed for this section. Socrata is the library for using sodapy, which will connect us to the data source

# In[1]:


import pandas as pd
from sodapy import Socrata
import datetime
import matplotlib.pyplot as plt
import numpy as np


# set up connection to data source. the app token is associated with an individual user's application, so mine is hidden. but they are free to obtain via data sfdata
# 
# the ID for the "fire-department-calls-for-service" dataset is 'nuek-vuh3' whihc sets up a connection to that dataset specifically

# In[2]:


domain = 'data.sfgov.org'
app_token = 'rrmNGKZrjY8MyQ07z8MxjVxHs'


# In[3]:


client = Socrata("data.sfgov.org", app_token, timeout = 30)
dataset = 'nuek-vuh3'


# let's start by making sure we're in the right place. we can obtain the metadata for the dataset using the get_metadata() method. then store the names of all the fields in a list, so we can pick for them later

# In[4]:



metadata = client.get_metadata(dataset)
columns = metadata.get('columns')
fieldNames = [col['fieldName'] for col in columns]
fieldNames


# fortunately/unfortunately, the API only lets a free user obtain 1000 per call. so in order to get all the data we want, we will need to make several calls and store the results in a list. we know we're going to need to do this a few times, so let's write a quick function for it. 

# In[5]:


def getAll(domain, app_token, dataset,fieldList,limit):
    
    client = Socrata(domain, app_token, timeout = 30)    
    results = list()
    
    for i in range(limit):        
        incoming = client.get(dataset, select = fieldList, offset = i*1000)
        results.append(incoming)
       
        if len(incoming)<1000:
            break
        pctComplete = 100* (len(results)/limit)
        if pctComplete % 10==0:
            print('percent complete :',pctComplete,  ' %')
            
    return results


# to get it working, let's define a fieldList and run it. recall that the list of fields we are selecting matches the list of fields that the course has chosen as the ones most pertinent to making predictions of response time.

# In[6]:


fieldList = 'received_dttm, dispatch_dttm, response_dttm, call_Type, fire_prevention_district, neighborhoods_analysis_boundaries, number_of_alarms, original_priority, priority, unit_type, rowid'


# In[7]:


results = getAll(domain,app_token,dataset,fieldList,limit=100)


# inconveniently, the results are stored in a list, with each element being a JSON string with 1000 records in it. so we need a function to loop through that and make a dataframe out of all the records

# In[8]:


def jsonToDF(json_results):
    results_df = pd.DataFrame()
    for i in range(len(json_results)):
        
        results_df= results_df.append(pd.DataFrame.from_records(json_results[i]), sort = False)
    return results_df
            


# In[9]:


results_df = jsonToDF(results) 
results_df


# because we didn't specify it, pandas assumes string objects for all columns in the dataframe. so now let's explicitly tell it what data type each column should be

# In[10]:


dataTypeDict = {'received_dttm':'datetime64',
                'dispatch_dttm':'datetime64',
                'response_dttm':'datetime64',
                'call_Type':'str',
                'fire_prevention_district':'str',
                'neighborhoods_analysis_boundaries':'str',
                'number_of_alarms':'int',
                'original_priority':'str',
                'priority':'str',
                'unit_type':'str'}    
results_df = results_df.astype(dataTypeDict)


# now we can do some preliminary math on the dataframe to get the target variable. recall that in the course, they called this "timeDelay" and defined it as "response time" - "received time". i have decided to call this "response duration" as i think it is a more descriptive and accurate name for the variable. i'll also calculate a field called "travel time" defined as "dispatch_dttm" - "response_dttm". just for fun. also, pandas will automatically store these as the "timedelta" datatype. which i think would be fine, but for simplicity of the ML model, let's convert them to minutes (as floats)

# In[11]:


results_df['response duration'] = results_df['response_dttm'] - results_df['received_dttm']
results_df['response duration'] = results_df['response duration']/np.timedelta64(1,'s')/60


# In[12]:


results_df['travel time'] = results_df['response_dttm'] - results_df['dispatch_dttm']
results_df['travel time'] = results_df['travel time']/np.timedelta64(1,'s')/60


# In[13]:


results_df.dtypes
        


# to get a sense of the distribution of the data we're dealing with, let's have a look at a histogram

# In[14]:


plt.hist(results_df['response duration'], bins = 30)


# looks like a pretty skewed distribution, with most values between 0 and 20 minutes. and a few much longer. so let's do as the course does and only focus on the values between 0 and 15 minutes, assuming the others are outliers that will hurt more than help our model.
# 
# i acknowledge that sometimes outliers are significant and we want our model to be able to learn from the fact that they are there, so i may remove this restriction later.

# In[15]:


temp = results_df[results_df['response duration'].between(0,15)]
plt.hist(temp['response duration'], bins = 30)


# with this filtering applied, we now a very pretty distribution, with a healthy mean right around 3 minutes

# this concludes the data import section. we will now store the results_df variable so we can use it in the next notebook, which cleans the data, explores it, and builds the models

# In[ ]:


# %store results_df

