"""
Created by Marcel V. Nguemaha, Dec, 2018.
"""

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

def getColumnDataTypes(data):
    
    # Numeric/Continuous
    num_cols = list(data._get_numeric_data().columns)

    # Categorical/discrete
    include=['object', 'category']
    cat_d = data.select_dtypes(include=include)
    cat_cols = list(cat_d.columns)
   
    # Date-time
    include=['datetime', 'datetime64','timedelta', 'timedelta64']
    dat_d = data.select_dtypes(include=include)
    date_cols = list(dat_d.columns)

    t, n, c, d = len(data.columns), len(num_cols), len(cat_cols), len(date_cols)
    
    # Check that we have all columns covered
    if n+c+d == t:
        pass
    else:
        print("Some columns NOT accounted for...")
        print ("Total columns: {}\nNumeric columns: {}\nCategorical columns: {}\nDatetime columns: {}".format(t, n, c, d))

    return (num_cols, cat_cols, date_cols)

class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    Select columns from a dataframe according to datatypes.

    See pandas.DataFrame.select_dtypes(include=[], exclude=[]):

    For all numeric types use the numpy dtype numpy.number
    For strings you must use the object dtype
    For datetimes, use np.datetime64, ‘datetime’ or ‘datetime64’
    For timedeltas, use np.timedelta64, ‘timedelta’ or ‘timedelta64’
    For Pandas categorical dtypes, use ‘category’
    For Pandas datetimetz dtypes, use ‘datetimetz’, or a ‘datetime64[ns, tz]’ 

    Notes:
        This class inherits the fit_transform from TransformerMixin

    args:
        pandas dataframe
    returns:
        subset for the requested dtype converted to numpy array.
    """

    def __init__(self, data_type=None):
        if data_type == None:
            self.data_type = [np.number]
        self.data_type = data_type
        self.attribute_names = None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        x_num = X.select_dtypes(include = self.data_type)
        self.attribute_names = list(x_num.columns)
        
        return x_num.values



class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        Columns of other types are imputed with median of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)
