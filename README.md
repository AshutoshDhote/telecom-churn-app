# 📉 Telecom Churn Prediction Web App

This is an interactive Streamlit web app that allows users to:
- Predict telecom customer churn using a machine learning model
- View key churn metrics and insights
- Explore sample customer inputs
- Visualize a Power BI dashboard
- See customers most likely to churn or stay

---

## 🚀 Features

- 🧠 Churn prediction based on customer, service, and billing info
- ⚖️ Most likely churners and loyal customers based on confidence
- 📈 Embedded Power BI Report
- 📌 Key Metrics Panel for churn stats
- 📋 Sample Inputs for testing and exploring predictions

---

## 🔧 Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/telecom-churn-app.git
cd telecom-churn-app

2. Install Dependencies

python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py

##🏃 Running the App
streamlit run app.py


## 📁 Project Structure
telecom-churn-app/
├── app.py                      # Streamlit app
├── models/
│   └── churn_model.pkl         # Trained ML model
├── data/
│   └── Customer_Data.csv       # Input dataset for form options
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
