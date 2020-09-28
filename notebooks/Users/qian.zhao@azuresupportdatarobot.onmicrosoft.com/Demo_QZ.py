# Databricks notebook source
# pylint: disable-all
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer

import pickle
import pandas as pd
import numpy as np


# Train/test split
df = pd.read_csv("/dbfs/FileStore/shared_uploads/qian.zhao@azuresupportdatarobot.onmicrosoft.com/Bank_Churn_Postcode.csv")
y = df.pop('Churn')
X_train, X_test, y_train, y_test = train_test_split(df, y)

numeric_features = list(X_train.select_dtypes(include=np.number).columns.values)
text_features = ['customerFeedback']
categorical_features = list(set(X_train.columns) - set(numeric_features + text_features))

#X_train = X_train[text_features].fillna(' ')

# Set up preprocessing steps for each type of feature
text_preprocessing = Pipeline(steps=[
    ('TfIdf', TfidfVectorizer())])

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        #('text1', text_preprocessing,text_features[0]),
        #('text2', text_preprocessing,text_features[1]),
        #('text3', text_preprocessing,text_features[2])
        ])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('RF', RandomForestClassifier())])


# COMMAND ----------

# Parameters of pipelines can be set using ‘__’ separated parameter names:
param_grid = [
        {#'RF__bootstrap': [False, True],
         'RF__n_estimators': [10],
         'RF__max_features': [0.6, 0.8],
         'RF__min_samples_leaf': [3, 5],
         'RF__min_samples_split': [3, 5]
        },
    ]

grid = GridSearchCV(pipeline, cv=5, n_jobs=1,param_grid=param_grid, iid=False,verbose=5)

grid.fit(X_train, y_train)

print("Best parameter (CV score=%0.3f):" % grid.best_score_)
print(grid.best_params_)

# COMMAND ----------

grid.cv_results_

# COMMAND ----------

#import joblib
#joblib.dump(grid.best_estimator_, '../sklearn/custom_model.pkl')


pkl_filename = "/dbfs/FileStore/shared_uploads/qian.zhao@azuresupportdatarobot.onmicrosoft.com/pickle_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(grid.best_estimator_, file)

# COMMAND ----------

