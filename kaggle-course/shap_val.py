#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap

seed = 1

data = pd.read_csv('/home/nicktheisen/data/kaggle/fifa-2018-match-statistics/FIFA 2018 Statistics.csv')
y = (data['Man of the Match'] == "Yes") # yes/no -> true/false

feature_names = [i for i in data.columns if data[i].dtype in [np.int64, np.int64]]
X = data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=seed)
rf_model = RandomForestClassifier(n_estimators=100, random_state=seed).fit(train_X, train_y)

row_to_analyse = 5
pred_data = val_X.iloc[row_to_analyse] # could also be multiple rows
pred_data_arr = pred_data.values.reshape(1, -1) # convert to np and reshape
print(rf_model.predict_proba(pred_data_arr))

# calculate shap values
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(pred_data)

# visualize shap values as force plot
plot = shap.force_plot(explainer.expected_value[1], shap_values[1], pred_data, matplotlib=False)
shap.save_html("force_plot.html", plot)

# visualize shap values as summary plot
shap_values = explainer.shap_values(val_X)
shap.summary_plot(shap_values[1], val_X)

# visualize shap values as dependece contribution plots
shap_values = explainer.shap_values(X)
shap.dependence_plot('Ball Possession %', shap_values[1], X, interaction_index='Goal Scored')
