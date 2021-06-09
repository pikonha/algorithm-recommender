
# %% 

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns

# %%

df = pd.read_csv('../data/training.csv', nrows=1000)
df.head()

# %%

features = df.iloc[:, 1:75]
target = df.iloc[:, 75]

features
target

# df = features.join(target)
# df

# %%
def find_correlation(data, threshold=0.9, remove_negative=False):
    """
    Given a numeric pd.DataFrame, this will find highly correlated features,
    and return a list of features to remove.
    Parameters
    -----------
    data : pandas DataFrame
        DataFrame
    threshold : float
        correlation threshold, will remove one of pairs of features with a
        correlation greater than this value.
    remove_negative: Boolean
        If true then features which are highly negatively correlated will
        also be returned for removal.
    Returns
    --------
    select_flat : list
        listof column names to be removed
    """
    corr_mat = data.corr()
    if remove_negative:
        corr_mat = np.abs(corr_mat)
    corr_mat.loc[:, :] = np.tril(corr_mat, k=-1)
    already_in = set()
    result = []
    for col in corr_mat:
        perfect_corr = corr_mat[col][corr_mat[col] > threshold].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            result.append(perfect_corr)
    select_nested = [f[1:] for f in result]
    select_flat = [i for j in select_nested for i in j]
    return select_flat

# %%

# columns_to_remove = find_correlation(df)
# columns_to_remove = ["sub_grade", *columns_to_remove]
# numeric_vars=[col for col in df.drop(columns_to_remove, axis=1).columns if col != 'int_rate']
# len(columns_to_remove)

# %%
# df.drop(columns_to_remove, axis=1)

# %%

X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=1)

# %%

import sklearn.metrics
import autosklearn.regression

# %%

automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=120,
    per_run_time_limit=30,
    tmp_folder='/tmp/autosklearn_regression_example_tmp',
    output_folder='/tmp/autosklearn_regression_example_out',
)

automl.fit(features, target)

# %%
print(automl.show_models())


# %%

predictions = automl.predict(X_test)
print("R2 score:", sklearn.metrics.r2_score(y_test, predictions))



# %%
# corr = df_full.corr()
# ax = sns.heatmap(corr,
#             xticklabels=corr.columns.values,
#             yticklabels=corr.columns.values)

# ax.set_title("Attributes Correlation Matrix", fontsize = 16)

# %%
# corr_target = abs(corr["int_rate"])
# corr_target

# #Selecting highly correlated features
# relevant_features = corr_target[corr_target>0.5]
# relevant_features

# %%
# print(df[["grade","sub_grade"]].corr())





# %%

# df_test = pd.read_csv('../data/test.csv')
# df_test.head()

# # %% 
# features_test = df_test.iloc[:, 1:75]
# target_test = df_test.iloc[:, -1:]

# target_test



# from pycaret import classification as clf

# # %%

# ml_setup = clf.setup(
#   data=df_full, 
#   target="int_rate", 
#   train_size=0.8,
#   session_id=1234,
#   numeric_features=numeric_vars
# )

# # %%
# clf.compare_models(fold=5)

# # %%
# regression_results = clf.pull()
# regression_results

# # %%
# model_metadata = clf.models()
# model_metadata['Name'] # with ids