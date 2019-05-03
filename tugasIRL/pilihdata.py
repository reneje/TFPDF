import pandas as pd 
import re

df = pd.read_csv('dataIRL2.csv')

dataframe1 = df[(df['tags'].str.contains('2019'))]
dataframe2 = df[(df['tags'].str.contains('KPK',flags=re.I))]
frame = [dataframe1,dataframe2]
result = pd.concat(frame)
result.to_csv("cobadataspesifik.csv",index=False)