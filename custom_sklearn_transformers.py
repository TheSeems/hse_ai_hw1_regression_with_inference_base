from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FunctionTransformer


def extract_value_from_unit_value(value):
    if value is None:
        return value
    if isinstance(value, float):
        return value
    splited = value.strip().split()
    if len(splited) == 1:
        return 0
    return float(splited[0])


def convert_to_int_feature(value):
    if value is None:
        return value
    return int(value)


class ColumnRemover(FunctionTransformer):
    def __init__(self, column_names: list[str]):
        self.column_names = column_names
        super().__init__(self._drop)

    def _drop(self, X):
        return X.drop(self.column_names, axis=1)


class IntFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, raw_cols: list[str]):
        self.medians = {}
        self.raw_cols = raw_cols

    def fit(self, X, y=None):
        for col in self.raw_cols:
            self.medians[col] = X[X[col].notna()][col].map(convert_to_int_feature).median()

    def transform(self, X):
        for col in self.raw_cols:
            medians = self.medians[col]
            X[col] = X[col].fillna(medians)
            X[col] = X[col].map(convert_to_int_feature).astype(int)
        return X


class ValueFromValueUnitExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, raw_cols: list[str]):
        self.medians = {}
        self.raw_cols = raw_cols

    def fit(self, X, y=None):
        for col in self.raw_cols:
            self.medians[col] = X[X[col].notna()][col].map(extract_value_from_unit_value).median()

    def transform(self, X):
        for col in self.raw_cols:
            medians = self.medians[col]
            X[col] = X[col].fillna(medians)
            X[col] = X[col].map(extract_value_from_unit_value).astype(float)
        return X


class FeatureBinarySplitter(BaseEstimator, TransformerMixin):
    def __init__(self, from_col: str, into_col_1: str, into_col_2: str):
        self.medians = {}
        self.from_col = from_col
        self.into_col_1 = into_col_1
        self.into_col_2 = into_col_2

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        X[[self.into_col_1, self.into_col_2]] = X[self.from_col].str.split(n=1, expand=True)
        X.drop([self.from_col], axis=1, inplace=True)
        return X
