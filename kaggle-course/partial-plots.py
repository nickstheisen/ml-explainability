#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

seed = 1

data = pd.read_csv('/home/nicktheisen/data/kaggle/fifa-2018-match-statistics/FIFA 2018 Statistics.csv')
y = (data['Man of the Match'] == "Yes") # yes/no -> true/false

feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]
X = data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=seed)
tree_model = DecisionTreeClassifier(random_state=seed, 
                                    max_depth=5, 
                                    min_samples_split=5).fit(train_X, train_y)

# export decision tree as pdf
tree_graph = tree.export_graphviz(tree_model, out_file=None, feature_names=feature_names)
s = graphviz.Source(tree_graph)
s.view()

# create and plot partial dependence for one feature
features_to_plot = ['Goal Scored', 'Distance Covered (Kms)']
disp = PartialDependenceDisplay.from_estimator(tree_model, val_X, features_to_plot)
plt.show()

# do the same for a Random Forest Model
rf_model = RandomForestClassifier(random_state=seed).fit(train_X, train_y)

disp_rf = PartialDependenceDisplay.from_estimator(rf_model, val_X, features_to_plot)
plt.show()

# plot partial dependence of in 2D
features_to_plot_2d = [('Goal Scored', 'Distance Covered (Kms)')]
disp_2d = PartialDependenceDisplay.from_estimator(tree_model, val_X, features_to_plot_2d)
plt.show()
