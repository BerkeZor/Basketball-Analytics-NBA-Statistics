import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import numpy as np
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle



def get_data():
    url = 'simple_data.csv'
    return pd.read_csv(url)
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
df = get_data()
df['position'] = df['position'].str.extract(r'(\w+),?')

X = df.drop(columns=['all_star'])

X = columnTransformer.fit_transform(X)
y = df.all_star

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)

model = LGBMClassifier()
model.fit(X_train, y_train)

pickle.dump(model, open('lgb.sav', 'wb'))
predictions = model.predict(X_test)

print("XGBoost Training Accuracy")
print(f'Accuracy: {round(accuracy_score(y_test, predictions) * 100, 3)}%')
print(classification_report(y_test, predictions))