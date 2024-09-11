from GetData import get_data
import pandas as pd
import os.path

path = r'C:\Users\fabaz\OneDrive\Escritorio\Code\Files'
filename = 'co2_data.csv'
url = 'https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_annmean_mlo.csv'    
get_data(path, filename, url)
df1 = pd.read_csv(os.path.join(path, filename), skiprows = range(0,43)) 
df1 = df1.drop(axis = 0, columns = 'unc')
df1 = df1.loc[:, ['year', 'mean']].groupby('year').mean()
df1

