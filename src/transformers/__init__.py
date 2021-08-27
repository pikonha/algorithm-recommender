import pandas as pd
from sklearn.preprocessing import LabelEncoder

def Dropna(X):
  return X.dropna(subset=["text_content", "feature_types"])

def DropColumns(X):
  return X.drop(columns=["reference","url", "feature_types", "text_content"], axis=1, errors='ignore')

def GetDummies(X):
  if "algorithm" in X.columns:
    X["algorithm"] = LabelEncoder().fit_transform(X["algorithm"])
  return X

def SplitColumn(X):
  r = []
  for row in X["feature_types"].str.split(',').tolist():
      c = {f'feature_{col}': 1 for col in row}
      r.append(c)

  feature_types = pd.DataFrame(r).fillna(0).astype(int)
  
  return pd.merge(
      feature_types.reset_index(drop=True),
      X.reset_index(drop=True),
      right_index=True, left_index=True
  ) 