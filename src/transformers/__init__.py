import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

class Dropna(BaseEstimator, TransformerMixin):

    def __init__(self, columns) -> None:
        self.columns = columns

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X.dropna(subset=self.columns)

class DropColumns(BaseEstimator, TransformerMixin):

    def __init__(self, columns) -> None:
        self.columns = columns

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns, axis=1)

class SplitColumn(BaseEstimator, TransformerMixin):

    def __init__(self, column) -> None:
        self.column = column

    def fit(self, X, y):
        return self

    def transform(self, X):
        r = []
        for row in X[self.column].str.split(',').tolist():
            c = {f'feature_{col}': 1 for col in row}
            r.append(c)

        feature_types = pd.DataFrame(r).fillna(0).astype(int)
        
        return pd.merge(
            feature_types.reset_index(drop=True),
            X.reset_index(drop=True),
            right_index=True, left_index=True
        ) 

class ModifiedLabelEncoder(LabelEncoder):

    def fit(self, _, y):
        super().fit(y)
        return self

    def fit_transform(self, _, y):
        return super().fit_transform(y).reshape(-1, 1)

    def transform(self, _, y):
        return super().transform(y).reshape(-1, 1)