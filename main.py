import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Cleaning Data
df = pd.read_csv("melb_data.csv")
df = df.drop(["Address", "SellerG", "Postcode", "CouncilArea", "Bedroom2"], axis=1)
df.to_csv("melb_data_dropped.csv", index=False)
#creating a dict of all suburbs for later viewing
subDic = {
    suburb: i
        for i, suburb in enumerate(df['Suburb'].unique())
    }
df['Suburb'] = df['Suburb'].map(subDic)
#filling in missing values for YearBuilt with the median of the suburb
df['YearBuilt'] = df['YearBuilt'].fillna(
    df.groupby('Suburb')['YearBuilt'].transform('median')
)
#bin the year built to the nearest half decade
df['YearBuilt'] = (df['YearBuilt'] // 5) * 5


df['landSize'] = df['landSize'].replace(0, np.nan)

#fill in missing value for LandSize with the median of the suburb, type and year built 
df['landSize'] = df['landSize'].fillna(
    df.groupby(['Suburb', 'YearBuilt', 'Type'])['landSize'].transform('median')
)
#fill in missing value for buildingArea using ratio of buildingArea to landSize for the suburb, type and year built
buildingAreaRatio = df.groupby(['Suburb', 'YearBuilt', 'Type']).apply(
    lambda x: 
        (x['buildingArea'] / x['landSize']).median()
)
df['buildingArea'] = df.apply(
    lambda row: 
        row['landSize'] * buildingAreaRatio[row['Suburb'], row['YearBuilt'], row['Type']] 
        if pd.isnull(row['buildingArea'])
        else row['buildingArea'], 
        axis=1
)
