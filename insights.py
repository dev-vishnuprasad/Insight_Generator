from flask import Flask, render_template, request, jsonify
import os
import json
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import base64

app = Flask(__name__)

# Configure Gemini
genai.configure(api_key="AIzaSyAUL9bvFucQbB0kJIuPS9RNvrr_CQ-YaY8")
model = genai.GenerativeModel('gemini-1.5-pro')


def get_image_data(file_path):
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

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