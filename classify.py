import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
import xgboost
from preprocess_script import target_column

def handle_missing_values(df):
    # Separate numerical and categorical columns
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = df.select_dtypes(include=['category']).columns
    print(categorical_columns)
    print(numeric_columns)
    # Impute numerical columns with median
    num_imputer = SimpleImputer(strategy='median')
    df[numeric_columns] = num_imputer.fit_transform(df[numeric_columns])

    # Impute categorical columns with most frequent value
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_columns] = cat_imputer.fit_transform(df[categorical_columns])

    return df

#Normalizes numerical values 
def normalize_numerical(train_df, test_df):
    scaler = StandardScaler()
    numerical_columns = train_df.select_dtypes(include=['int64', 'float64']).columns
    
    train_df[numerical_columns] = scaler.fit_transform(train_df[numerical_columns])
    test_df[numerical_columns] = scaler.transform(test_df[numerical_columns])
    
    return train_df, test_df

#Enocodes categorical features
def encode_categorical(df):
    oe = OrdinalEncoder()
    le = LabelEncoder()
    
    for column in df.select_dtypes(include=['category', 'object']).columns:
        if column != target_column:
            # Reshape the column to a 2D array
            df[column] = oe.fit_transform(df[[column]])
        else:
            df[column] = le.fit_transform(df[column])
    
    return df,le


def classify(test_path,train_path,output_path):
    test_data = pd.read_csv(test_path)
    train_data = pd.read_csv(train_path)




if __name__ == "__main__":

    classify('preprocessed/preprocessed_test_data.csv', 'preprocessed/preprocessed_train_data.csv', 'model/')
