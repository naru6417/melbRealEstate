import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler,  OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import pipeline    
from sklearn.compose import ColumnTransformer


# Cleaning Data
df = pd.read_csv("melb_data.csv")
df = df.drop(["Address", "SellerG", "Postcode", "CouncilArea", "Bedroom2"], axis=1)
df.to_csv("melb_data_dropped.csv", index=False)

df['Landsize'] = df['Landsize'].replace(0, np.nan)
df['BuildingArea'] = df['BuildingArea'].replace(0, np.nan)

#Split data into training and testing sets

x = df.drop('Price', axis=1)
y = df['Price']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

#Year built imputation logic with fallback
yearbuilt_suburb_median = x_train.groupby('Suburb')['YearBuilt'].median()
yearbuilt_global_median = x_train['YearBuilt'].median()
    

#car imputation logic with fallback
car_suburb_yearbuilt_type_median = x_train.groupby(['Suburb', 'YearBuilt', 'Type'])['Car'].median()
car_suburb_type_median = x_train.groupby(['Suburb', 'Type'])['Car'].median()
car_suburb_median = x_train.groupby(['Suburb'])['Car'].median()
car_global_median = x_train['Car'].median()


#land size imputation logic with fallback
landsize_suburb_yearbuilt_type_median = x_train.groupby(['Suburb', 'YearBuilt', 'Type'])['Landsize'].median()
landsize_suburb_type_median = x_train.groupby(['Suburb', 'Type'])['Landsize'].median()
landsize_suburb_median = x_train.groupby(['Suburb'])['Landsize'].median()
landsize_global_median = x_train['Landsize'].median()


#generate building area ratio for building area imputation with fallback with known values 
known_values = x_train[
    x_train['BuildingArea'].notna() & x_train['Landsize'].notna () & (x_train['Landsize'] > 0)
].copy()

known_values['BuildingAreaRatio'] = known_values['BuildingArea'] / known_values['Landsize']

ratio_suburb_yearbuilt_type = known_values.groupby(['Suburb', 'YearBuilt', 'Type'])['BuildingAreaRatio'].median()
ratio_suburb_type = known_values.groupby(['Suburb', 'Type'])['BuildingAreaRatio'].median()
ratio_suburb = known_values.groupby(['Suburb'])['BuildingAreaRatio'].median()
global_ratio = known_values['BuildingAreaRatio'].median()

def impute_data( 
    data, 
    yearbuilt_suburb_median, 
    yearbuilt_global_median, 
    car_suburb_yearbuilt_type_median, 
    car_suburb_type_median, 
    car_suburb_median, 
    car_global_median, 
    landsize_suburb_yearbuilt_type_median, 
    landsize_suburb_type_median, 
    landsize_suburb_median, 
    landsize_global_median,
    ratio_suburb_yearbuilt_type,
    ratio_suburb_type,
    ratio_suburb,
    global_ratio
):
    data = data.copy()
    #Fill missing values for YearBuilt
    data['YearBuilt'] = data['YearBuilt'].fillna(data['Suburb'].map(yearbuilt_suburb_median))
    data['YearBuilt'] = data['YearBuilt'].fillna(yearbuilt_global_median)
    data['YearBuilt'] = (data['YearBuilt'] // 5) * 5
    #Fill missing values for Car
    data['Car'] = data['Car'].fillna(data.set_index(['Suburb', 'YearBuilt', 'Type']).index.map(car_suburb_yearbuilt_type_median))
    data['Car'] = data['Car'].fillna(data.set_index(['Suburb', 'Type']).index.map(car_suburb_type_median))
    data['Car'] = data['Car'].fillna(data['Suburb'].map(car_suburb_median))
    data['Car'] = data['Car'].fillna(car_global_median)
    #Fill missing values for Landsize
    data['Landsize'] = data['Landsize'].fillna(data.set_index(['Suburb', 'YearBuilt', 'Type']).index.map(landsize_suburb_yearbuilt_type_median))
    data['Landsize'] = data['Landsize'].fillna(data.set_index(['Suburb', 'Type']).index.map(landsize_suburb_type_median))
    data['Landsize'] = data['Landsize'].fillna(data['Suburb'].map(landsize_suburb_median))
    data['Landsize'] = data['Landsize'].fillna(landsize_global_median)
    #Fill missing values for BuildingArea
    data['BuildingArea'] = data['BuildingArea'].fillna(data.set_index(['Suburb', 'YearBuilt', 'Type']).index.map(ratio_suburb_yearbuilt_type * data['Landsize']))
    data['BuildingArea'] = data['BuildingArea'].fillna(data.set_index(['Suburb', 'Type']).index.map(ratio_suburb_type * data['Landsize']))
    data['BuildingArea'] = data['BuildingArea'].fillna(data['Suburb'].map(ratio_suburb * data['Landsize']))
    data['BuildingArea'] = data['BuildingArea'].fillna(global_ratio * data['Landsize'])

    return data

cat_cols = x.select_dtypes(include=['object']).columns
num_cols = x.select_dtypes(include=['int64', 'float64']).columns

#Linear Regression 
linearPre = ColumnTransformer(
    transformers=[
        (
            'num',
            StandardScaler(),
            num_cols
        ),
        (
            'cat',
            OneHotEncoder(handle_unknown='ignore'),
            cat_cols
            
        )
    ]  
)

LinearReg = pipeline(
    steps = [
        ('preprocessor', linearPre),
        ('model', LinearRegression())
])

#Random Forest
forestPre = ColumnTransformer(
    transformers = [
        (
            'nums',
            'passthrough',
            num_cols
        )
        (
            'cat',
            OneHotEncoder(handle_unknown= 'ignore'),
            cat_cols
        )
    ]
)
RandomForest = pipeline(
    steps = [
        ('preprocessor', forestPre)
        ('model', RandomForestRegressor())
    ]
)
LinearReg.fit(x_train, y_train)

print(df[cat_cols].isna().sum())
print(df[['Landsize', 'BuildingArea']].isna().sum())