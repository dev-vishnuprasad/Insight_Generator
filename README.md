<p align="center">
  <img src="https://img.icons8.com/?size=100&id=12175&format=png&color=000000" alt="Income Classification Logo" width="100"/>
</p>

<h1 align="center">Insight generation Web App 📊✨</h1>
<p align="center">
  <strong>A Flask-powered tool to preprocess, model, and visualize income data with style! 🚀</strong>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python" alt="Python Version"/>
  <img src="https://img.shields.io/badge/License-GPL%20v3-green?style=flat-square" alt="License"/>
  <img src="https://img.shields.io/badge/Flask-Web%20App-orange?style=flat-square&logo=flask" alt="Flask"/>
</p>

---

## 🌟 What’s This All About?
Welcome to the **Insight generation Web App**! This sleek Flask-based application lets you upload CSV datasets, preprocess them like a pro, train powerful machine learning models (RandomForest & XGBoost), and generate stunning visualizations and AI-driven insights using Gemini. Whether you're a data scientist or a curious explorer, this tool makes income prediction fun and insightful! 🎉

## 🚀 Features That Shine
- 📂 **Upload with Ease**: Drop your train and test CSV files and get started.
- 🛠️ **Preprocessing Magic**: Clean, encode, and deduplicate data with customizable scripts.
- 🤖 **Model Mastery**: Train RandomForest and XGBoost with GridSearchCV for top performance.
- 📈 **Visual Vibes**: Create feature importance plots, histograms, and correlation heatmaps.
- 💡 **AI Insights**: Let Gemini AI uncover deep insights from your data.
- 💾 **JSON Goodies**: Save model results and distributions in neat JSON files.

## 🛠️ Prerequisites
- Python 3.8+ 🐍
- Libraries: Flask, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, Joblib, Google Generative AI (Gemini), Werkzeug 📚

## ⚡ Quick Start Guide
Get up and running in minutes! Follow these steps:

1. **Clone the Repo**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Gemini API**:
   Set your Gemini API key 🔑:
   ```bash
   export gemini_key='your-gemini-api-key'  # Windows: set gemini_key=your-g #$ % gemini_key
   ```

5. **Create Directories**:
   ```bash
   mkdir -p uploads preprocessed output model
   ```

6. **Launch the App**:
   ```bash
   python app.py
   ```
   Open `http://127.0.0.1:5000` in your browser 🌐 and start exploring!

## 🎮 How to Use
1. **Upload Data**: Navigate to the homepage and upload your train and test CSV files 📤.
2. **Preprocess**: Tweak the preprocessing script (if needed) and clean your data ✏️.
3. **Train Models**: Generate plots and train models with a single click 📊.
4. **Explore Insights**: Dive into Gemini’s AI-generated insights and beautiful visualizations 👀.

**Data Tip**: The default script expects columns like `age`, `workclass`, `education`, `income`, etc. Check `preprocess_script.py` to customize for your dataset 📝.

## 🗂️ Project Structure
```
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
```

## 📝 Notes
- The `classify.py` script is a work-in-progress and not fully integrated 🚧.
- Ensure your Gemini API key is valid 🔐.
- Default target column is `income`. Update `preprocess_script.py` for custom datasets 🔍.
- All plots and JSON files are saved in the `output/` directory 💾.

## 🤝 Contribute
Love this project? Join the fun! Fork the repo, create a branch, and submit a pull request. Make sure your changes align with the GNU GPL v3 license 🌟.

## 📜 License
This project is proudly licensed under the **GNU General Public License v3**. See the [LICENSE](LICENSE) file for details.

## 📬 Get in Touch
Got questions or ideas? Open an issue on the repo or reach out to the maintainer. Let’s make data science sparkle! ✨

---
<p align="center">
  Built with ❤️ by data enthusiasts
</p>
