import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
from xgboost import XGBClassifier
from preprocess_script import target_column

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

def normalize_numerical(train_df, test_df):
    scaler = StandardScaler()
    numerical_columns = train_df.select_dtypes(include=['int64', 'float64']).columns
    
    train_df[numerical_columns] = scaler.fit_transform(train_df[numerical_columns])
    test_df[numerical_columns] = scaler.transform(test_df[numerical_columns])
    
    return train_df, test_df


def handle_missing_values(df):
    # Separate numerical and categorical columns
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns

    # Handle numerical missing values
    if len(numerical_columns) > 0:
        num_imputer = SimpleImputer(strategy='mean')
        df[numerical_columns] = num_imputer.fit_transform(df[numerical_columns])

    # Handle categorical missing values
    if len(categorical_columns) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_columns] = cat_imputer.fit_transform(df[categorical_columns])

    return df

def train_and_save_models(train_path,test_path, output_dir, target_column):
    # Load and preprocess data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    train_data = handle_missing_values(train_data)
    test_data = handle_missing_values(test_data)

    # Normalize numerical features
    train_data, test_data = normalize_numerical(train_data, test_data)
    # Load and preprocess data
    train_data,le = encode_categorical(train_data)
    test_data,le = encode_categorical(test_data)

    # Separate features and target for train and test
    X_train = train_data.drop('income', axis=1)
    original_feature_names = X_train.columns.tolist()
    print(original_feature_names)
    X_train = X_train.values
    y_train = train_data['income'].values
    X_test = test_data.drop('income', axis=1).values
    y_test = test_data['income'].values
    
    # Define models and parameter grids
    models = {
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3]
            }
        }
    }
    
    results = {}
    
    for model_name, model_info in models.items():
        # Perform grid search
        grid_search = GridSearchCV(model_info['model'], model_info['params'], cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Make predictions and calculate accuracy
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save model
        model_path = os.path.join('', 'model/', f'{model_name}.joblib')
        joblib.dump(best_model, model_path)
        
        # Get feature importances
        if model_name == 'RandomForest':
            importances = best_model.feature_importances_
        else:
            best_model.get_booster().feature_names = original_feature_names  # XGBoost
            importances = best_model.feature_importances_
            feature_importance = best_model.get_booster().get_score(importance_type='weight')

            # Sort feature importances in descending order
            sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

            # Create a list of dictionaries for JSON
            json_data = [{"feature": k, "importance": v} for k, v in sorted_importance]

            # Save as JSON
            with open(os.path.join(output_dir, 'xgboost_feature_importances.json'), 'w') as f:
                json.dump(json_data, f, indent=2)

            # If you still want to save the plot as an image
            best_model.get_booster().feature_names = original_feature_names
            xgb.plot_importance(best_model)
            plt.savefig(os.path.join(output_dir, 'xgboost_feature_importances.png'))
        if model_name == 'RandomForest':    
            feature_importances = pd.Series(importances, index=original_feature_names).sort_values(ascending=False)
            
            # Plot feature importances
            plt.figure(figsize=(10, 6))
            feature_importances.plot(kind='bar')
            plt.title(f'{model_name} Feature Importances')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model_name}_feature_importances.png'))
            plt.close()

            
            
            # Save feature importances as JSON
            with open(os.path.join(output_dir, f'{model_name}_feature_importances.json'), 'w') as f:
                json.dump(feature_importances.to_dict(), f, indent=4)
        
        # Store results
        results[model_name] = {
            'accuracy': accuracy,
            'best_params': grid_search.best_params_
        }
    
    # Save accuracies and best parameters as JSON
    with open(os.path.join(output_dir, 'model_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    return 'Models trained and saved successfully'


def json_serializable(obj):
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'Object of type {type(obj)} is not JSON serializable')


def objToCat(df):
    df_copy = df.copy()
    for col in df_copy.columns:
        # Check if the column is of object dtype
        if df_copy[col].dtype == 'object':
            df_copy[col] = df_copy[col].astype('category')
    return df_copy

def plot_and_save_numeric_distributions(df, numeric_column, target_column, output_dir):
    categories = df[target_column].unique()
    
    plt.figure(figsize=(12, 6))
    
    json_data = {}
    
    for category in categories:
        category_data = df.loc[df[target_column] == category, numeric_column].dropna()
        
        if category_data.empty:
            print(f"Skipping empty category {category} for column {numeric_column}")
            continue
        
        try:
            hist, bin_edges = np.histogram(category_data, bins=20, density=True)
            plt.hist(category_data, bins=20, density=True, alpha=0.5, label=f'{target_column}={category}')
        
            json_data[str(category)] = {
                'histogram': hist.tolist(),
                'bin_edges': bin_edges.tolist(),
                'statistics': {
                    'mean': category_data.mean(),
                    'median': category_data.median(),
                    'std': category_data.std(),
                    'min': category_data.min(),
                    'max': category_data.max()
                }
            }
        except ValueError as e:
            print(f"Error processing category {category} for column {numeric_column}: {str(e)}")
            continue
    
    if not json_data:
        print(f"Skipping plot for column {numeric_column} due to lack of valid data")
        return
    
    plt.title(f'Relative Frequency Distribution of {numeric_column}')
    plt.xlabel(numeric_column)
    plt.ylabel('Density')
    plt.legend()
    
    plt.tight_layout()
    
    plot_filename = f'{numeric_column}_relative_distribution.png'
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.close()
    
    json_filename = f'{numeric_column}_relative_distribution_data.json'
    with open(os.path.join(output_dir, json_filename), 'w') as json_file:
        json.dump(json_data, json_file, indent=4, default=json_serializable)

def plot_and_save_categorical_distributions(df, categorical_column, target_column, output_dir):
    categories = df[target_column].unique()
    
    plt.figure(figsize=(12, 6))
    
    json_data = {}
    
    for category in categories:
        category_data = df.loc[df[target_column] == category, categorical_column].dropna()
        
        if category_data.empty:
            print(f"Skipping empty category {category} for column {categorical_column}")
            continue
        
        value_counts = category_data.value_counts(normalize=True)
        plt.bar(value_counts.index, value_counts.values, alpha=0.5, label=f'{target_column}={category}')
        
        json_data[str(category)] = {
            'values': value_counts.index.tolist(),
            'frequencies': value_counts.values.tolist(),
            'statistics': {
                'mode': category_data.mode().tolist(),
                'unique_count': category_data.nunique()
            }
        }
    
    if not json_data:
        print(f"Skipping plot for column {categorical_column} due to lack of valid data")
        return
    
    plt.title(f'Relative Frequency Distribution of {categorical_column}')
    plt.xlabel(categorical_column)
    plt.ylabel('Relative Frequency')
    plt.legend()
    
    plt.tight_layout()
    
    plot_filename = f'{categorical_column}_relative_distribution.png'
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.close()
    
    json_filename = f'{categorical_column}_relative_distribution_data.json'
    with open(os.path.join(output_dir, json_filename), 'w') as json_file:
        json.dump(json_data, json_file, indent=4, default=json_serializable)

def generate_plots_and_json(df, target_column, output_dir):
    for column in df.columns:
        if column != target_column:
            if df[column].dtype.name in ['float64', 'int64']:
                if df[column].isnull().all():
                    print(f"Skipping numeric column {column} as it contains only NaN values")
                    continue
                plot_and_save_numeric_distributions(df, column, target_column, output_dir)
            elif df[column].dtype.name in ['object', 'category']:
                if df[column].isnull().all():
                    print(f"Skipping categorical column {column} as it contains only NaN values")
                    continue
                plot_and_save_categorical_distributions(df, column, target_column, output_dir)
            else:
                print(f"Skipping column {column} with dtype {df[column].dtype.name}")

def generate_plots(test_path,train_path,output_dir):
    test_data = pd.read_csv(test_path)
    train_data = pd.read_csv(train_path)

    Processed_data = pd.concat([train_data,test_data])
    Processed_data = objToCat(Processed_data)

    categorical_columns = list(Processed_data.select_dtypes(include=['category']).columns)
    numerical_columns = list(Processed_data.select_dtypes(include=['number']).columns)


    #******************************************************************************#


    for colname in categorical_columns:
        plt.figure(figsize=(10, 6))  
        plt.title('Column: ' + colname)
        
        # Get value counts
        value_counts = Processed_data[colname].value_counts().head(20)
        
        # Plot
        value_counts.plot(kind='barh', color="#1f77b4")
        
        # Save plot
        plot_filename = f'{colname}_plot.png'
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, plot_filename))
        plt.close()  # Close the plot to free up memory
        
        # Store data for JSON
        json_data = value_counts.to_dict()
        
        # Save JSON data
        json_filename = f'{colname}_data.json'
        with open(os.path.join(output_dir, json_filename), 'w') as json_file:
            json.dump(json_data, json_file, indent=4)


    #******************************************************************************#
    
    
    for column_name, column_series in Processed_data.select_dtypes(include=['number']).items():
        plt.figure(figsize=(10, 5))
        plt.title('Column: ' + column_name)
        
        # Create histogram
        hist, bin_edges = np.histogram(column_series.dropna(), bins=30)
        
        # Plot histogram
        plt.hist(column_series.dropna(), bins=30, edgecolor='black')
        plt.xlabel(column_name)
        plt.ylabel('Frequency')
        
        # Save plot
        plot_filename = f'{column_name}_histogram.png'
        plt.savefig(os.path.join(output_dir, plot_filename))
        plt.close()  

        # Prepare data for JSON
        json_data = {
            'histogram': hist.tolist(),
            'bin_edges': bin_edges.tolist(),
            'statistics': {
                'mean': column_series.mean(),
                'median': column_series.median(),
                'std': column_series.std(),
                'min': column_series.min(),
                'max': column_series.max()
            }
        }

        # Save JSON data
        json_filename = f'{column_name}_histogram_data.json'
        with open(os.path.join(output_dir, json_filename), 'w') as json_file:
            json.dump(json_data, json_file, indent=4, default=json_serializable)


    #******************************************************************************#
    
    #defining correlation between only numerical columns
    corr = Processed_data.select_dtypes(include=['number']).corr()
    base_filename = "heatmap_correlation"

    print("heatmap")
    # Generate the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr, 
        xticklabels=corr.columns.values,
        yticklabels=corr.columns.values,
        annot=True,  
        cmap='coolwarm', 
        vmin=-1, vmax=1  
    )
    plt.title('Correlation Heatmap')
    print("triple kill")
    plt.tight_layout()
    print("double heatmp")
    # Save the heatmap image
    heatmap_filename = f"{base_filename}.png"
    plt.savefig(os.path.join('output/', heatmap_filename))
    plt.close()  # Close the plot to free up memory

    print(f"Heatmap saved as {heatmap_filename}")

    # Create JSON representation of the heatmap data
    heatmap_data = []
    for i, row in enumerate(corr.index):
        for j, col in enumerate(corr.columns):
            heatmap_data.append({
                'x': col,
                'y': row,
                'value': corr.iloc[i, j]
            })

    # Save the JSON to a file in output folder
    json_filename = f"{base_filename}.json"
    with open(os.path.join('output/', json_filename), 'w') as f:
        json.dump(heatmap_data, f, indent=2)

    print(f"JSON data saved as {json_filename}")

    #******************************************************************************#

    generate_plots_and_json(Processed_data, target_column, output_dir)

    train_and_save_models(train_path,test_path,output_dir,target_column)

    return 'success'


if __name__ == "__main__":

    generate_plots('preprocessed/preprocessed_test_data.csv', 'preprocessed/preprocessed_train_data.csv', 'output/')
