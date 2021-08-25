#%%
n_rows = int(input("n_rows"))
n_features = int(input("n_features"))
problem_type_classification = int(input("problem_type_classification"))
problem_type_regression = int(input("problem_type_regression"))
target_type_continuous = int(input("target_type_continuous"))
target_type_discrete = int(input("target_type_discrete"))
type_of_learning_supervised = int(input("type_of_learning_supervised"))

#%%
import pandas as pd

#%%
model = pd.read_pickle("../models/decision.tree.pkl")
model

#%%
data = pd.DataFrame({
  "n_rows": [n_rows],
  "n_features": [n_features],
  "problem_type_classification": [problem_type_classification],
  "problem_type_regression": [problem_type_regression],
  "target_type_continuous": [target_type_continuous],
  "target_type_discrete": [target_type_discrete],
  "type_of_learning_supervised": [type_of_learning_supervised],
}, columns=model["columns"])
data = data.fillna(0)

#%%
model["model"].predict(data)
