import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Cleaning Data
df = pd.read_csv("melb_data.csv")
df = df.drop(["Address", "Rooms", "SellerG", "Postcode", "CouncilArea", "Bedroom2"], axis=1)
df.to_csv("melb_data_dropped.csv", index=False)