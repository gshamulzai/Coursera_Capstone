#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup
import pandas as pd
import requests
import json
import numpy as np


# In[50]:


#Retrieving wiki page as text to parse in html parser 
website_url = requests.get('https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M').text
soup = BeautifulSoup(website_url, 'html.parser')
table = soup.find("table", { "class" : "wikitable sortable"})

#setting the values in the table to a column
A=[]
B=[]
C=[]


for row in table.find_all('tr'):
    cells = row.find_all('td')
    if len(cells)==3: #Only extract table body not heading
        A.append(cells[0].find(text=True))
        B.append(cells[1].find(text=True))
        C.append(cells[2].find(text=True))
        

df=pd.DataFrame(A,columns=["Postcode"])
df["Borough"]=B
df["Neighbourhood"]=C
df = df.reindex_axis(['Postcode', 'Borough', 'Neighbourhood'], axis=1)
df2 = df.copy()
                
df2.merge(df, on=['Postcode'])

df2['Neighborhood'] = df['Neighbourhood'].astype(str)+','+df2['Neighbourhood'].astype(str)

df2 = df2[df2.Borough != 'Not assigned']
df2 = df2.reindex_axis(['Postcode', 'Borough', 'Neighborhood'], axis=1)

df2.shape
df2


# In[ ]:




