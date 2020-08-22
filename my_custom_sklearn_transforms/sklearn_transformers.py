from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')
    
    
def my_mean_imputer( df, column, mean ):
    
    data = df[ column ].values
    
    if data.ndim > 1:
        data = data[:,0]
        
    n_data = len( data )

    find_nan = np.isnan( data )

    corrected = [ mean if find_nan[i] else data[i] for i in range(n_data) ]
    
    df.drop( columns = column )
    
    df[ column ] = corrected
        
    return df


def get_mean( df, column ):

    data = df[ column ].values
    
    if data.ndim > 1:
        data = data[:,0]
    
    mean = np.nanmean( data )
    
    return mean


class MeanColumnImputer(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        data = X.copy()
        self.mean = get_mean( data, column = self.column )
        return self
    
    def transform(self, X):
        data = X.copy()
        return my_mean_imputer(data, column = self.column, mean = self.mean)
    
 
def my_zero_imputer( df, exception = ["PERFIL"] ):
    
    features = df.columns.values
    
    for ft in features:
        
        if len( set( [ ft ] ).intersection( exception ) ) == 0:

            data = df[ ft ].values
            
            if data.ndim > 1:
                data = data[:,0]
            
            n_data = len( data )

            find_nan = np.isnan( data )

            if( np.sum( find_nan ) != 0 ):

                corrected = [ 0 if find_nan[i] else data[i] for i in range(n_data) ]

                df[ ft ] = corrected

    return df



class ZeroImputer(BaseEstimator, TransformerMixin):
    def __init__(self, exception):
        self.exception = exception

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe 
        return my_zero_imputer(data, exception=self.exception)
    
    
def feature_engineering( df ):
    
    df = pd.DataFrame.from_records( data = df )
    
    n_data = len( df.index )
    
    MF = df["NOTA_MF"].values
    DE = df["NOTA_DE"].values
    EM = df["NOTA_EM"].values
    GO = df["NOTA_GO"].values
    
    REP_MF = df["REPROVACOES_MF"].values
    REP_DE = df["REPROVACOES_DE"].values
    REP_EM = df["REPROVACOES_EM"].values
    REP_GO = df["REPROVACOES_GO"].values
    
    MEDIA = ( MF + DE + EM + GO )/4
    df["MEDIA"] = MEDIA
    
    df["DE"] = DE/10 + REP_DE/3
    df["EM"] = EM/10 + REP_EM/3
    df["MF"] = MF/10 + REP_MF/3
    df["GO"] = GO/10 + REP_GO/3
    
    sqrt_MF = np.sqrt( MF )
    sqrt_DE = np.sqrt( DE )
    sqrt_EM = np.sqrt( EM )
    sqrt_GO = np.sqrt( GO )
    df["SQRT_MF"] = sqrt_MF
    df["SQRT_DE"] = sqrt_DE
    df["SQRT_EM"] = sqrt_EM
    df["SQRT_GO"] = sqrt_GO
    
    tan_MF = np.tan( MF ) 
    tan_DE = np.tan( DE ) 
    tan_EM = np.tan( EM ) 
    tan_GO = np.tan( GO ) 
    df["TAN_MF"]  = tan_MF
    df["TAN_DE"]  = tan_DE
    df["TAN_EM"]  = tan_EM
    df["TAN_GO"]  = tan_GO
    
    return df



class CustomFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        None

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return feature_engineering( data )    
    
