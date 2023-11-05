#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import GridSearchCV
from time import time
import warnings
import pickle

# Data preparation
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
df.rename(columns={'DEATH_EVENT': 'death_event'}, inplace=True)
for col in ['anaemia', 'diabetes', 'high_blood_pressure', 'smoking']:
    df[col].replace(to_replace=[0, 1], value=['No', 'Yes'], inplace=True)
df.sex.replace(to_replace=[0, 1], value=['Female', 'Male'], inplace=True)
t0 = time()

# Data spliting and training
X = df.drop('death_event', axis=1)
y = df.death_event
X_full_train, X_test, y_full_train, y_test = train_test_split(X, y, test_size=0.2,
                                                              stratify=y,
                                                              random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_full_train, y_full_train, 
                                                  stratify=y_full_train,
                                                  test_size=0.25, random_state=1)

X_full_train = X_full_train.reset_index(drop=True)
y_full_train = y_full_train.reset_index(drop=True)
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
X_val = X_val.reset_index(drop=True)
y_val = y_val.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


# One-hot encoding
def data_transformation(X_train, X_val, X_test):
    global dv
    X_train_dicts = X_train.to_dict(orient='records')
    X_val_dicts = X_val.to_dict(orient='records')
    X_test_dicts = X_test.to_dict(orient='records')
    dv = DictVectorizer(sparse=True)
    X_cat_tr = dv.fit_transform(X_train_dicts)
    feature_names = list(dv.get_feature_names_out())
    X_cat_val = dv.transform(X_val_dicts)
    X_cat_test = dv.transform(X_test_dicts)
    return X_cat_tr, X_cat_val, X_cat_test, feature_names


X_cat_train, X_cat_val, _, feature_names = data_transformation(X_train, X_val, X_test)


# Modeling with Logistic Regression, Decision Trees and Random Forests
with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)

    lr = LogisticRegression(max_iter=1000)
    dt = DecisionTreeClassifier(random_state=1)
    rf = RandomForestClassifier(random_state=1)

    lr.fit(X_cat_train, y_train)
    dt.fit(X_cat_train, y_train)
    rf.fit(X_cat_train, y_train)

    y_pred_lr = lr.predict(X_cat_val)
    y_pred_dt = dt.predict(X_cat_val)
    y_pred_rf = rf.predict(X_cat_val)

    auc_result_lr = roc_auc_score(y_val, y_pred_lr)
    auc_result_dt = roc_auc_score(y_val, y_pred_dt)
    auc_result_rf = roc_auc_score(y_val, y_pred_rf)

    f1_score_lr = f1_score(y_val, y_pred_lr)
    f1_score_dt = f1_score(y_val, y_pred_dt)
    f1_score_rf = f1_score(y_val, y_pred_rf)

print(f'Testing {lr.__class__.__name__}, {dt.__class__.__name__} and {rf.__class__.__name__}:')
print()
print(f'F1 score using {lr.__class__.__name__}: {f1_score_lr.round(3):>17}')
print(f'AUC using {lr.__class__.__name__}: {auc_result_lr.round(3):>22}')
print()
print(f'F1 score using {dt.__class__.__name__}: {f1_score_dt.round(3):>13}')
print(f'AUC using {dt.__class__.__name__}: {auc_result_dt.round(3):>18}')
print()
print(f'F1 score using {rf.__class__.__name__}: {f1_score_rf.round(3):>13}')
print(f'AUC using {rf.__class__.__name__}: {auc_result_rf.round(3):>18}')


# Parameter tuning with Random Forests
param_grid = [{'n_estimators': [50, 100, 200],
              'max_depth': [2, 5, 10, 15],
              'min_samples_leaf': [2, 5, 10, 15]},]

with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    rf_orig = clone(rf)
    grid_search = GridSearchCV(rf_orig, param_grid, cv=5,
                               scoring='neg_log_loss')
    grid_search.fit(X_cat_train, y_train)

print()
print(f'Best parameters for {rf.__class__.__name__}: {grid_search.best_params_}')
print()

with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)

    best_estimator_rf = grid_search.best_estimator_
    best_estimator_rf.fit(X_cat_train, y_train)
    y_pred_rf = best_estimator_rf.predict(X_cat_val)
    auc_result_rf = roc_auc_score(y_val, y_pred_rf)
    f1_score_rf = f1_score(y_val, y_pred_rf)

print(f'F1 score using best {rf.__class__.__name__} estimator: {f1_score_rf.round(3):>8}')
print(f'AUC using best {rf.__class__.__name__} estimator: {auc_result_rf.round(3):>13}')


# Use all of the data to fit the default RF model
X_full_train_dicts = X_full_train.to_dict(orient='records')
X_full_train_transfored = dv.fit_transform(X_full_train_dicts)
X_test_dicts = X_test.to_dict(orient='records')
X_test_transfored = dv.transform(X_test_dicts)

with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)

    rf.fit(X_full_train_transfored, y_full_train)
    y_pred_rf = rf.predict(X_test_transfored)
    auc_result_rf = roc_auc_score(y_test, y_pred_rf)
    f1_score_rf = f1_score(y_test, y_pred_rf)

print()
print(f'F1 score using default {rf.__class__.__name__} estimator with training and validation dataset: {f1_score_rf.round(3):>8}')
print(f'AUC using default {rf.__class__.__name__} estimator with training and validation dataset: {auc_result_rf.round(3):>13}')


# Saving best model to pickle file
best_model_file = 'model.pkl'
with open(best_model_file, "wb") as f:
    pickle.dump(rf, f)

# Saving DictVectorizer model to pickle file
dv_file = 'dv.pkl'
with open(dv_file, "wb") as f:
    pickle.dump(dv, f)

print()
print(f'Saved {rf.__class__.__name__} in {best_model_file} file.')
print(f'Saved {dv.__class__.__name__} class instance in {dv_file} file.')
print()
print(f'Time elapsed: {time() - t0:.3f} seconds.')
