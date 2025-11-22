from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
import pandas as pd
import joblib

def normalizar(df, scaler):
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    dfNormalized = scaler.fit_transform(df[num_cols])
    dfNormalized = pd.DataFrame(dfNormalized, columns=num_cols, index=df.index)
    joblib.dump(scaler, "./data/scaler.pkl")
    return dfNormalized

def OHE(df, column_name):
    try:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop=None)
    except TypeError:
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

    ohe_mat = ohe.fit_transform(df[[column_name]])
    ohe_cols = ohe.get_feature_names_out([column_name])
    df_ohe = pd.DataFrame(ohe_mat, columns=ohe_cols, index=df.index)
    return df_ohe

def le_encode(df, column_name):
    le = LabelEncoder()
    df[column_name] = le.fit_transform(df[column_name])
    return df[column_name]