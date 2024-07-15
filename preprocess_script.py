# Your default preprocessing script here
import pandas as pd

#specifying the target column
    #Edit this for custom dataset
target_column = "income"

#Removes duplicates between test and train data
#Do not Edit for custom datasets

def rmDuplicates(train_data, test_data):
    train_data['is_test'] = 0
    test_data['is_test'] = 1

    df = pd.concat([train_data, test_data])
    df = df.drop_duplicates()
    train_data = df[df.is_test == 0].drop('is_test', axis=1)
    test_data = df[df.is_test == 1].drop('is_test', axis=1)
    
    return train_data, test_data

#Converts object type into category
#Do not Edit for custom datasets

def objToCat(df):
    df_copy = df.copy()
    for col in df_copy.columns:
        # Check if the column is of object dtype
        if df_copy[col].dtype == 'object':
            df_copy[col] = df_copy[col].astype('category')
            
            
    return df_copy


def csvToDf(path):

    #Specifying column names and their types

    #Edit this if you have a custom dataset
    data_types = {
        "age": "int",
        "workclass": "category",
        "fnlwgt": "int",  
        "education": "category",
        "education_num": "int",
        "marital_status": "category",
        "occupation": "category",
        "relationship": "category",
        "race": "category",
        "sex": "category",
        "capital_gain": "float",  
        "capital_loss": "int",
        "hours_per_week": "int",
        "native_country": "category",
        "income": "category",
    }
    column_names = ['age', 'workclass', 'fnlwgt', 'education','education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain','capital_loss', 'hours_per_week', 'native_country', 'income']
    #column_names = list(data_types.keys())
    
    df = pd.read_csv(
        path,
        names=column_names,
        index_col=None,
        skipinitialspace=True,  # Skip spaces after delimiter
        #This converts custom NaN values into numpy NaN values
        #Edit this for custom dataset
        na_values={

            'capital_gain': 99999,
            'workclass': '?',
            'native_country': '?',
            'occupation': '?',

        },
        dtype = data_types,
    )

    #These are Extra preprocessing steps

    #Edit this for custom dataset



    #removing the fnlwgt column
    df = df.drop('fnlwgt', axis=1)

    #changing the datatype of the colums from object to category
    df = df.astype({col: 'category' for col in df.select_dtypes(['object'])})

    #removing the dot found in the income column
    df['income'] = df.income.str.rstrip('.').astype('category')

    #deleting duplicates
    df = df.drop_duplicates()
    return df

def preprocess_data(test_data_path, train_data_path, output_path):

    # Read the CSV files
    test_data = csvToDf(test_data_path)
    train_data = csvToDf(train_data_path)

    train_data, test_data = rmDuplicates(train_data, test_data)

    test_data = objToCat(test_data)
    train_data = objToCat(train_data)
    # Save the preprocessed data
    test_data.to_csv(output_path + 'preprocessed_test_data.csv', index=False)
    train_data.to_csv(output_path + 'preprocessed_train_data.csv', index=False)

# Don't execute anything when the script is imported

if __name__ == "__main__":

    preprocess_data('uploads/test_data.csv', 'uploads/train_data.csv', 'preprocessed/')

