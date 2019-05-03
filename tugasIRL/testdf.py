(import pandas as pd 

df = pd.read_csv('dataIRL1.csv')

baru = df.drop_duplicates(subset=['tag'], keep=False)

baru.to_csv("dataIRL2.csv"))