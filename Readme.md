🏠 House Price Prediction using Machine Learning
This project was developed as part of the Machine Learning course at M. S. Ramaiah Institute of Technology. It was a team effort involving two members, focusing on building a full machine learning pipeline for predicting house prices using real-world data.

📌 Overview
This system predicts house prices based on user-input features like the number of bedrooms, square footage, etc. We experimented with multiple regression models:
✅ Gradient Boosting Regressor (Best performer – 99.86% accuracy)
Random Forest Regressor
XGBoost
Ridge Regression
Linear Regression
The dataset used is the King County House Sales dataset (USA).
For deeper insights, we’ve also included a research paper analyzing our model performance and methodology.

🚀 Getting Started
🔧 Requirements
Python 3.x
Required packages listed in requirements.txt

💻 Run Locally
Step 1: Clone the repository
git clone https://github.com/Krishna-S-27/House_Price_Prediction.git
cd House_Price_Prediction

Step 2: Install dependencies
pip install -r requirements.txt

Step 3: Train the model (if model file is missing)
python model_training.py

Step 4: Run the Streamlit app
streamlit run app.py

This will open a local web interface in your browser where you can enter house features and get an instant price prediction.

📄 Research Paper
A detailed research paper is included in the repository. It outlines our data preprocessing, feature selection, model tuning, and results comparison. We recommend reviewing it for a complete understanding of our approach.

📬 Contact
If you have any questions or need help running the project, feel free to reach out:

📧 Email: krishnashalawadi27@gmail.com

💻 GitHub: Krishna-S-27

📷 Instagram: @itskrrisshhh
