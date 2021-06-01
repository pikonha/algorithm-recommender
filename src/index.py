
# %% 

import pandas as pd
from sklearn import pipeline
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
# %%

features.join(target)

# %%

corr = features.join(target).corr() 

ax = sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

ax.set_title("Attributes Correlation Matrix", fontsize = 16)
