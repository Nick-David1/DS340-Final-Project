import pandas as pd

# read accepted file
df = pd.read_csv('data/sofa data.csv')

df["accepted"] = 1

print(df.head())