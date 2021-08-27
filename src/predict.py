#%%
n_rows = int(input("n_rows"))
n_features = int(input("n_features"))
problem_type_classification = int(input("problem_type_classification"))
problem_type_regression = int(input("problem_type_regression"))
target_type_continuous = int(input("target_type_continuous"))
target_type_discrete = int(input("target_type_discrete"))
type_of_learning_supervised = int(input("type_of_learning_supervised"))

#%%
n_rows = 150
n_features = 5
problem_type = "regression"
# reference = "Iris Flower Dataset"
type_of_learning = "supervised"
# algorithm = "support vector machine"
feature_types = "continuous,binary"
target_type = "continuous"
text_content = "text_content"

#%%
import pandas as pd

#%%
model = pd.read_pickle("../models/pipeline.pkl")
model

#%%
data = pd.DataFrame({
  "n_rows": [n_rows],
  "n_features": [n_features],
  "problem_type": [problem_type],
  # "reference": [reference],
  "type_of_learning": [type_of_learning],
  # "algorithm": [algorithm],
  "feature_types": [feature_types],
  "target_type": [target_type],
  # "text_content": [text_content]
}, columns=model["features"] + ["algorithm"])
data = data.fillna(0)

data.head()

# %%
model["model"].transform(data)
