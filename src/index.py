
# %% 
import pandas as pd

# %%
df = pd.read_csv(
  '../data/text.csv',
  dtype={
    'reference': str,
    'problem_type': str,
    'n_rows': int,
    'n_columns': int,
    'target_type': str,
    'feature_type': str,
    'type_of_learning': str,
    'text_content': str,
    'algorithm': str,
    'URL': str
  }
)
df.head()

#%%
df.dropna(subset=["text_content", "feature_types"], inplace=True)
df.shape

# %%
df['algorithm'] = df['algorithm'].str.lower()
df[['algorithm', 'problem_type']].value_counts()

# %%
r = []
for row in df['feature_types'].str.split(',').tolist():
  c = {f'feature_{col}': 1 for col in row}
  r.append(c)
feature_types = pd.DataFrame(r).fillna(0).astype(int)
feature_types.head()

# %%
df.drop(columns=["reference","url", "feature_types"], axis=1, inplace=True)

# %%
continuous_features = df.iloc[:,1:3]
continuous_features
# %%
discrete_features = pd.merge(
  df.iloc[:,:1].reset_index(drop=True),
  df.iloc[:,3:-2].reset_index(drop=True),
  left_index=True, right_index=True
) 
discrete_features = pd.get_dummies(discrete_features)
discrete_features.head()
# %%
X = pd.merge(
  continuous_features.reset_index(drop=True), 
  discrete_features.reset_index(drop=True),
  left_index=True, right_index=True
) 
X.head()

# %%
target = df.iloc[:, -2]
y = target

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
X = pd.merge(
  X.reset_index(),
  df_tokens.reset_index(),
  left_index=True, right_index=True
)
X.head()

# %%  
# ===========================================================================#
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# %% 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# %% 
models = [
  GaussianNB(),
  DecisionTreeClassifier(),
  RandomForestClassifier(n_estimators=10, random_state=9, class_weight='balanced'),
  Perceptron(eta0=1, random_state=1),
  SVC(),
  LogisticRegression(),
  KNeighborsClassifier()
]
# %% 
index = []
model_bench = {
  "accuracy": [],
  "precision": [],
  "recall": [],
  "fscore": [],
}
for model in models:
  model.fit(X_train,y_train)
  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred,average='weighted')
  index.append(type(model).__name__)
  
  model_bench["accuracy"].append(accuracy)
  model_bench["precision"].append(precision)
  model_bench["recall"].append(recall)
  model_bench["fscore"].append(fscore)

model_bench
# %% 
pd.options.display.float_format = '{:.2%}'.format
bench_df = pd.DataFrame(model_bench, index=index)
bench_df.head(20)
