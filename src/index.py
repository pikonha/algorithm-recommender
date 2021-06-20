# %% 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

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
r = []
for row in df['feature_type'].str.split(',').tolist():
  c = {f'feature_{col}': 1 for col in row}
  r.append(c)

feature_types = pd.DataFrame(r).fillna(0).astype(int)
feature_types

# %%
df = df.drop(columns=["reference","URL", "feature_type"], axis=1)
df = pd.concat([feature_types, df], axis=1)
df.head()

# %%
df['algorithm'].value_counts()

# %%
from sklearn.preprocessing import LabelEncoder

# %%
features = df.iloc[:,:8]
encoded_features = pd.get_dummies(features)
encoded_features.head()

# %%
target = df.iloc[:, -1]
encoded_target = LabelEncoder().fit_transform(target)
encoded_target

# %% 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# %% 
X_train, X_test, y_train, y_test = train_test_split(encoded_features, encoded_target, test_size=1/3, random_state=4)

# %% 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score,LeaveOneOut
from numpy import mean
from numpy import std

# %%
cv = KFold(n_splits=10, random_state=1, shuffle=True)

# %%
model = DecisionTreeClassifier()
# model.fit(X_train, y_train)

# %%
scores = cross_val_score(model, encoded_features, encoded_target, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

# %%
for i in range(2,31):
  cv = KFold(n_splits=i, random_state=1, shuffle=True)
  model = DecisionTreeClassifier()
  scores = cross_val_score(model, encoded_features, encoded_target, scoring='accuracy', cv=cv, n_jobs=-1)
  print('%d, Accuracy: %.3f (%.3f)' % (i, mean(scores), std(scores)))


