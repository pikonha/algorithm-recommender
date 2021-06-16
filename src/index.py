
# %% 

from matplotlib.pyplot import axis
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns

# %%

df = pd.read_csv(
  '../data/problem_solutions.csv',
  dtype={
    'reference': str,
    'problem_type': str,
    'n_rows': int,
    'n_columns': int,
    'target_type': str,
    'feature_type': str,
    'type_of_learning': str,
    'algorithm': str,
    'URL': str
  }
)
df.head()

# %%
df.shape

# %%
from functools import reduce 

r = []
for row in df['feature_type'].str.split(',').tolist():
  c = {f'feature_{col}': 1 for col in row}
  r.append(c)

feature_types = pd.DataFrame(r).fillna(0)

# %%
df = df.drop(columns=["reference","URL", "feature_type"], axis=1)

# %%
df.info()

# %%
df['algorithm'].value_counts()


# %%
df.head()
# %%
features =  pd.concat([feature_types, df], axis=1)
features = features.iloc[:,:8]
y = df.iloc[:, -1]


# %%

X = pd.get_dummies(features)
X.head()


# %% 

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score
from numpy import mean
from numpy import std

# %%

kfold = KFold(n_splits=3, random_state=2, shuffle=True)

model = DecisionTreeClassifier()
model.fit(X, y)

scores = cross_val_score(model, X, y, scoring='accuracy', cv=kfold, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

# %%

from sklearn.cluster import KMeans

# %%
model = KMeans(n_clusters=4, random_state=1)
scores = cross_val_score(model, X, y, scoring='accuracy', cv=kfold, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

# %%

df_f = pd.get_dummies(df)
df_f

# %%
corr = df_f.corr()
ax = sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

ax.set_title("Attributes Correlation Matrix", fontsize = 16)


