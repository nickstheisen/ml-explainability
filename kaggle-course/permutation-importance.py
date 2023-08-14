'''
Code origin: https://www.kaggle.com/code/dansbecker/permutation-importance
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import eli5
from eli5.sklearn import PermutationImportance

seed = 1

data = pd.read_csv('/home/nicktheisen/data/kaggle/fifa-2018-match-statistics/FIFA 2018 Statistics.csv')
print(data['Man of the Match'])
y = (data['Man of the Match'] == "Yes") # yes/no -> true/false

feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]
X = data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=seed)
rf_model = RandomForestClassifier(n_estimators=100, random_state=seed).fit(train_X, train_y)

perm = PermutationImportance(rf_model, random_state=seed).fit(val_X, val_y)
print(eli5.format_as_text(eli5.explain_weights(perm, feature_names = val_X.columns.to_list())))
#eli5.show_weights(perm, feature_names = val_X.columns.to_list()) # only works with IPython
