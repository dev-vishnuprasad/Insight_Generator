
  


Insight Generation Web App 📊✨

  A Flask-powered tool to preprocess, model, and visualize any data with style! 🚀


  
  
  



🌟 What’s This All About?
Welcome to the Income Classification Web App! This sleek Flask-based application lets you upload CSV datasets, preprocess them like a pro, train powerful machine learning models (RandomForest & XGBoost), and generate stunning visualizations and AI-driven insights using Gemini. Whether you're a data scientist or a curious explorer, this tool makes income prediction fun and insightful! 🎉
🚀 Features That Shine

📂 Upload with Ease: Drop your train and test CSV files and get started.
🛠️ Preprocessing Magic: Clean, encode, and deduplicate data with customizable scripts.
🤖 Model Mastery: Train RandomForest and XGBoost with GridSearchCV for top performance.
📈 Visual Vibes: Create feature importance plots, histograms, and correlation heatmaps.
💡 AI Insights: Let Gemini AI uncover deep insights from your data.
💾 JSON Goodies: Save model results and distributions in neat JSON files.

🛠️ Prerequisites

Python 3.8+ 🐍
Libraries: Flask, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, Joblib, Google Generative AI (Gemini), Werkzeug 📚

⚡ Quick Start Guide
Get up and running in minutes! Follow these steps:

Clone the Repo:
git clone <repository-url>
cd <repository-directory>


Set Up a Virtual Environment:
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt


Configure Gemini API:Set your Gemini API key 🔑:
export gemini_key='your-gemini-api-key'  # Windows: set gemini_key=your-g #$ % gemini_key


Create Directories:
mkdir -p uploads preprocessed output model


Launch the App:
python app.py

Open http://127.0.0.1:5000 in your browser 🌐 and start exploring!


🎮 How to Use

Upload Data: Navigate to the homepage and upload your train and test CSV files 📤.
Preprocess: Tweak the preprocessing script (if needed) and clean your data ✏️.
Train Models: Generate plots and train models with a single click 📊.
Explore Insights: Dive into Gemini’s AI-generated insights and beautiful visualizations 👀.

Data Tip: The default script expects columns like age, workclass, education, income, etc. Check preprocess_script.py to customize for your dataset 📝.
🗂️ Project Structure
├── app.py                    🖥️ Main Flask application
├── config.py                 ⚙️ Target column config
├── preprocess_script.py      🛠️ Data preprocessing script
├── generate_plot.py          📈 Plot and model training script
├── classify.py               🚧 Classification script (WIP)
├── LICENSE                   📜 GNU GPL v3 License
├── uploads/                  📂 Uploaded CSV files
├── preprocessed/             📂 Preprocessed data
├── output/                   📂 Plots and JSON outputs
├── model/                    📂 Trained models
└── templates/                📄 HTML templates

📝 Notes

The classify.py script is a work-in-progress and not fully integrated 🚧.
Ensure your Gemini API key is valid 🔐.
Default target column is income. Update preprocess_script.py for custom datasets 🔍.
All plots and JSON files are saved in the output/ directory 💾.

🤝 Contribute
Love this project? Join the fun! Fork the repo, create a branch, and submit a pull request. Make sure your changes align with the GNU GPL v3 license 🌟.
📜 License
This project is proudly licensed under the GNU General Public License v3. See the LICENSE file for details.
📬 Get in Touch
Got questions or ideas? Open an issue on the repo or reach out to the maintainer. Let’s make data science sparkle! ✨


  Built with ❤️ by data enthusiasts
