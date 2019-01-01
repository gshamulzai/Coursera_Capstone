#!/usr/bin/env python
# coding: utf-8

# In[62]:


from bs4 import BeautifulSoup
import pandas as pd
import requests
import json
import numpy as np
import io


# In[108]:


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
        

df=pd.DataFrame(A,columns=["Postal_Code"])
df["Borough"]=B
df["Neighbourhood"]=C
df = df.reindex_axis(['Postal_Code', 'Borough', 'Neighbourhood'], axis=1)
df2 = df.copy()
                
df2.merge(df, on=['Postal_Code'])

df2['Neighborhood'] = df['Neighbourhood'].astype(str)+','+df2['Neighbourhood'].astype(str)

df2 = df2[df2.Borough != 'Not assigned']
df2 = df2.reindex_axis(['Postal_Code', 'Borough', 'Neighborhood'], axis=1)


postalcode = pd.read_csv("https://cocl.us/Geospatial_data/Geospatial_Coordinates.csv")
df3 = df2.copy()
df3['Latitude'] = postalcode['Latitude']
df3['Longitude'] = postalcode['Longitude']

df3 = df3.reindex_axis(['Postal_Code', 'Borough', 'Neighborhood', 'Latitude','Longitude' ], axis=1)



#df3.merge(postalcode,on =["Postal_Code"],how = "outer")

df3


# In[ ]:




