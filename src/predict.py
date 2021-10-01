#%%
# n_rows = int(input("n_rows"))
# n_features = int(input("n_features"))
# problem_type_classification = int(input("problem_type_classification"))
# problem_type_regression = int(input("problem_type_regression"))
# target_type_continuous = int(input("target_type_continuous"))
# target_type_discrete = int(input("target_type_discrete"))
# type_of_learning_supervised = int(input("type_of_learning_supervised"))

n_rows = 150
n_features = 5
problem_type = "regression"
target_type = "continuous"
type_of_learning = "supervised"
algorithm = "support vector machine"
feature_types = "continuous,binary"
text_content = "The Iris flower data set is a multivariate data set introduced by the British statistician and biologist Ronald Fisher in his 1936 paper The use of multiple measurements in taxonomic problems. It is sometimes called Anderson's Iris data set because Edgar Anderson collected the data to quantify the morphologic variation of Iris flowers of three related species. The data set consists of 50 samples from each of three species of Iris (Iris Setosa, Iris virginica, and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.This dataset became a typical test case for many statistical classification techniques in machine learning such as support vector machinesContentThe dataset contains a set of 150 records under 5 attributes - Petal Length, Petal Width, Sepal Length, Sepal width and Class(Species).AcknowledgementsThis dataset is free and is publicly available at the UCI Machine Learning Repository"

#%%
import pandas as pd

#%%
model = pd.read_pickle("../models/decision.tree.pkl")
model

#%%
df = pd.DataFrame({
  "n_rows": [n_rows],
  "n_features": [n_features],
  "problem_type": [problem_type],
  "target_type": [target_type],
  "type_of_learning": [type_of_learning],
  "feature_types": [feature_types],
  "text_content": [text_content]
})

# %%
r = []
for row in df['feature_types'].str.split(',').tolist():
  c = {f'feature_{col}': 1 for col in row}
  r.append(c)
feature_types = pd.DataFrame(r).fillna(0).astype(int)
feature_types.head()

# %%
df.drop(columns=["feature_types"], axis=1, inplace=True)
df.head()
# %%
continuous_features = df.iloc[:,:2]
continuous_features
# %%
discrete_features = df.iloc[:,2:5]
discrete_features = pd.get_dummies(discrete_features)
discrete_features.head()
# %%
features = pd.merge(
  continuous_features.reset_index(drop=True), 
  discrete_features.reset_index(drop=True),
  left_index=True, right_index=True
) 
features.head()
#%%
# ===========================================================================#

## TEXT PROCESSING
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter

nltk.download('stopwords')

#%%
stop_words = set(stopwords.words('english'))
porter = PorterStemmer()
def handle_text(text):
  tokens = word_tokenize(text)
  tokens = [
    porter.stem(w) for w in tokens 
    if not w in stop_words and len(w) > 2 and w.isalpha()]
  return ' '.join(tokens)

#%%
def get_most_frequent_tokens(tokens, limit=10):
  with_stp = Counter()
  with_stp.update(tokens)
  return [x for x,_ in with_stp.most_common(limit)]
  
#%%
# tokenize
tokens = df['text_content'].dropna().apply(handle_text)
tokens = tokens.apply(lambda x: x.split())
#%%
most_frequent_tokens = tokens.apply(lambda x: get_most_frequent_tokens(x, 15))
most_frequent_tokens
#%%
df_tokens = pd.DataFrame.from_records(most_frequent_tokens)
df_tokens.head(20)

# %%
w = []
for _, row in df_tokens.iterrows():
  w.append({col: 1 for col in row})

df_tokens = pd.DataFrame(w).fillna(0).astype(int)
df_tokens.head()

# %%
features = pd.merge(
  features.reset_index(),
  df_tokens.reset_index(),
  left_index=True, right_index=True
)
features.head()

#%%
model["model"].predict(features)
