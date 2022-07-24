import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  
import seaborn as sns
import plotly.express as px
import statsmodels.api as sm
import pickle
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report


# Read data
df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv', sep=',')


# Filter some strange observartions
df_filter = df_raw.copy()
df_filter = df_raw[(df_raw["BMI"] > 0 ) & (df_raw["BloodPressure"] > 0) & (df_raw["Glucose"] > 0)]


# Final df
df = df_filter.copy() 


# Split the dataset
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=28)

## Modeling

# Model 1

clf = DecisionTreeClassifier(criterion='entropy',random_state=0)
clf.fit(X_train, y_train)
print('Accuracy model 1 - test set:', clf.score(X_test, y_test))


# Model 2: decision tree + grid search of hyperparameters
params = {'criterion': ['gini', 'entropy'], 'max_depth': range(1,10), 'min_samples_split':range(1,10), 'min_samples_leaf': range(1,5)}
grid = GridSearchCV(DecisionTreeClassifier(random_state=0), params, verbose=1, n_jobs=-1,cv=3)
grid.fit(X_train, y_train)

model_cv = grid.best_estimator_
print('Accuracy of tree selected by CV in test set:',grid.score(X_test, y_test))


# Model 3: random forest
rfc = RandomForestClassifier(n_estimators=100, random_state=0)
rfc.fit(X_train, y_train)
print('Accuracy RF test data:',rfc.score(X_test, y_test))


# Model 4: random forest + grid search of hyperparameters
param_grid = [{'max_depth': [8, 12, 16], 
         'min_samples_split': [12, 16, 20], 
         'criterion': ['gini', 'entropy']}]

rfc=RandomForestClassifier(random_state=1107)
grid_rfc=GridSearchCV(estimator=rfc,param_grid=param_grid, cv=5, n_jobs=-1,verbose=2)
grid_rfc.fit(X_train,y_train)
model_rfc_cv=grid_rfc.best_estimator_

print('Accuracy of random forest selected by CV in test set (grid search):',grid_rfc.score(X_test, y_test))

## The classifier with highest accuracy is model 4

# Save final model

filename = '../models/final_model.sav'
pickle.dump(model_rfc_cv, open(filename, 'wb'))