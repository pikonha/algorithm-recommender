
# %% 

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns

# %%

df = pd.read_csv('../data/problem_solutions.csv')
df.head()

# %%
df.shape

# %%

df = df.drop(columns=["URL"], axis=1)

# %%

features = df.iloc[:,1:6]
y = df.iloc[:, 6]

# %%

X = pd.get_dummies(features)
X.head()

# %%

from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, cross_val_score
from numpy import mean
from numpy import std

# %%
kfold = KFold(n_splits=3, random_state=2, shuffle=True)

# %%
model = KMeans(n_clusters=4, random_state=1)
scores = cross_val_score(model, X, y, scoring='accuracy', cv=kfold, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

# %%

# df_f = pd.concat([X, y], axis=1)
df_f = pd.get_dummies(df)
df_f

# %%
corr = df_f.corr()
ax = sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

ax.set_title("Attributes Correlation Matrix", fontsize = 16)


