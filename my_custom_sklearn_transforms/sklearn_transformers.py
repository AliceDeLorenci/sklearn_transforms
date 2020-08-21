from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

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
    
def feature_engineering( df ):
    
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
    
    df["DE"] = DE + REP_DE
    df["EM"] = EM + REP_EM
    df["MF"] = MF + REP_MF
    df["GO"] = GO + REP_GO
    
    sqrt_MF = np.sqrt( MF )
    sqrt_EM = np.sqrt( EM )
    df["SQRT_MF"] = sqrt_MF
    df["SQRT_EM"] = sqrt_EM
    
    tan_MF = np.tan( MF ) 
    df["TAN_MF"]  = tan_MF
    
    return df    


class CustomFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        None

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe com as novas colunas
        return feature_engineering( data )
