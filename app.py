import os
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import subprocess
import time
from flask import Flask, request, render_template, send_file, redirect, url_for, flash, session,jsonify
from werkzeug.utils import secure_filename
import pandas as pd
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import base64

app = Flask(__name__)

# Configure Gemini
gemini_key = os.environ["gemini_key"]
genai.configure(api_key= gemini_key)
model = genai.GenerativeModel('gemini-1.5-pro')

app.secret_key = 'session_key'  

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
PREPROCESSED_FOLDER = os.path.join(os.path.dirname(__file__), 'preprocessed')
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREPROCESSED_FOLDER'] = PREPROCESSED_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREPROCESSED_FOLDER, exist_ok=True)

def get_image_data(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        uploaded_files = []
        for file_key in ['file1', 'file2']:
            if file_key not in request.files:
                flash(f'Please upload both CSV files. Missing {file_key}.')
                return redirect(request.url)
            
            file = request.files[file_key]
            
            if file.filename == '':
                flash(f'No selected file for {file_key}.')
                return redirect(request.url)
            
            if file and allowed_file(file.filename):
                filename = 'test_data.csv' if file_key == 'file1' else 'train_data.csv'
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                uploaded_files.append(file_path)
            else:
                flash(f'Invalid file type for {file_key}. Please upload CSV files only.')
                return redirect(request.url)
        
        session['uploaded_files'] = uploaded_files
        flash('Files uploaded successfully.')
        return redirect(url_for('preprocess'))
    
    return render_template('index.html')

@app.route('/preprocess', methods=['GET', 'POST'])
def preprocess():
    uploaded_files = session.get('uploaded_files', [])
    if len(uploaded_files) != 2:
        flash('Error: Two files were not uploaded.')
        return redirect(url_for('upload_files'))

    default_script = """
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

"""

    if request.method == 'POST':
        script = request.form['script']
        script_path = os.path.join(app.root_path, 'preprocess_script.py')
        with open(script_path, 'w') as f:
            f.write(script)
        
        try:
            # Execute the script
            exec(script, {'__name__': '__main__', 'pd': pd})
            flash('Preprocessing complete.')
            return redirect(url_for('generate_plot'))
        except Exception as e:
            flash(f'Error during preprocessing: {str(e)}')
            return redirect(url_for('preprocess'))
       
        
    else:
        script = default_script
        return render_template('edit_script.html', script=script)

@app.route('/generate_plot',methods =['GET','POST'])
def generate_plot():
    if request.method == 'POST':
        
        script_path = os.path.join(app.root_path, 'generate_plot.py')
        

        try:
            result = subprocess.run(['python', script_path], capture_output=True, text=True, check=False)
            print("Script output:")
            print(result.stdout)
            return redirect(url_for('combine_json'))
        except subprocess.CalledProcessError as e:
            print("Script error (CalledProcessError):")
            print(e.stderr)
            return 'Script execution error', 500  # Return an error response
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return f'Unexpected error: {str(e)}', 500 
    else:
        print("hello")
        return render_template('generate_plot.html')
    
@app.route('/combine_json')
def combine_json():
    folder_path = 'output/'  # Replace with your folder path
    output_file = os.path.join(folder_path, 'combined_json.txt')
    
    with open(output_file, 'w') as outfile:
        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(folder_path, filename)
                outfile.write(f"File: {filename}\n")
                
                with open(file_path, 'r') as json_file:
                    json_content = json.load(json_file)
                    json.dump(json_content, outfile, indent=2)
                
                outfile.write('\n\n')
    
    return redirect(url_for('generate_insights'))


@app.route('/generate_insights', methods=['GET'])
def generate_insights():
    output_dir = 'output/'
    with open('output/combined_json.txt') as file:
        context = file.read()
    # Generate insights using Gemini
    prompt = f"""
    The following information is multiple json files combined together.Their filename is specified at the start of the each file.
    The following json files are created by processing a dataset.The dataset is subjeted to classification and feature imporance 
    is also specified.

    

    {context}

    Please provide:
    1. A comprehensive summary of the model performance, key features, and data distributions.
    2. Detailed insights into the most important features and their potential impact on the target variable (income).
    3. Analysis of the distribution of numerical and categorical features, including any notable patterns or anomalies.
    4. Suggestions for further analysis or improvement of the models based on the feature distributions.
    5. Potential business implications based on the feature importances, model performance, and data distributions.
    6. Any potential biases or limitations in the dataset that might affect the model's performance or interpretability.

    Please structure your response in a clear, concise manner, using bullet points or numbered lists where appropriate.
    """

    response = model.generate_content(
        prompt,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }
    )

    insights = response.text

    # Prepare image data for the template
    images = {}
    for file in os.listdir(output_dir):
        if file.endswith('.png'):
            images[file] = get_image_data(os.path.join(output_dir, file))

    return render_template('insights.html', insights=insights, images=images)
if __name__ == '__main__':
        app.run(debug=True)
