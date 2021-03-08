# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 12:09:44 2021

@author: zgp21
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

autompg = pd.read_csv('auto-mpg.data', delim_whitespace = True, 
                      names = ['mpg', 'cylinders','displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name'], 
                      index_col = 'car_name').replace(to_replace = '?', value = np.NaN)
autompg['horsepower'] = autompg['horsepower'].astype(float)
autompg = autompg.fillna(autompg.mean())
autompg = autompg.values

x = autompg[:,1:]
y = autompg[:,0]

### BASELINE MODEL

def baseline_model():
    model = Sequential()
    # Add hidden layer with same number of neurons as predictors
    model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
    # Add output layer
    model.add(Dense(1, kernel_initializer='normal'))
    # Optimize MSE with ADAM method
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

estimator = KerasRegressor(build_fn = baseline_model, epochs = 50, batch_size = 5, verbose = 0)
kfold = KFold(n_splits = 10)
results = cross_val_score(estimator, x, y, cv = kfold)
print(abs(results.mean()))

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)
results = cross_val_score(pipeline, x, y, cv=kfold)
print(abs(results.mean()))

# define the model
def larger_model():
	# create model
	model = Sequential()
	model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)
results = cross_val_score(pipeline, x, y, cv=kfold)
print(abs(results.mean()))

# define wider model
def wider_model():
	# create model
	model = Sequential()
	model.add(Dense(20, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=wider_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10)
results = cross_val_score(pipeline, x, y, cv=kfold)
print(abs(results.mean()))