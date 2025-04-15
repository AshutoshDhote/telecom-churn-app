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

##Here's a breakdown of the columns in CSV dataset:

Customer_ID: Unique identifier for each customer.

Gender: Gender of the customer (e.g., "Male", "Female").

Age: The age of the customer.

Married: Whether the customer is married (e.g., "Yes", "No").

State: The state or region where the customer resides.

Number_of_Referrals: The number of referrals made by the customer to others.

Tenure_in_Months: The number of months the customer has been with the company.

Value_Deal: The value of the deal the customer has, likely indicating some sort of contract or agreement.

Phone_Service: Whether the customer has phone service (e.g., "Yes", "No").

Multiple_Lines: Whether the customer subscribes to multiple phone lines (e.g., "Yes", "No").

Internet_Service: Whether the customer has internet service (e.g., "Yes", "No").

Internet_Type: Type of internet service the customer has (e.g., "Fiber Optic", "DSL", etc.).

Online_Security: Whether the customer has online security as part of their plan (e.g., "Yes", "No").

Online_Backup: Whether the customer has online backup services (e.g., "Yes", "No").

Device_Protection_Plan: Whether the customer has a device protection plan (e.g., "Yes", "No").

Premium_Support: Whether the customer has premium support (e.g., "Yes", "No").

Streaming_TV: Whether the customer has streaming TV service (e.g., "Yes", "No").

Streaming_Movies: Whether the customer has streaming movies service (e.g., "Yes", "No").

Streaming_Music: Whether the customer has streaming music service (e.g., "Yes", "No").

Unlimited_Data: Whether the customer has unlimited data (e.g., "Yes", "No").

Contract: The type of contract the customer is under (e.g., "Month-to-Month", "One Year", "Two Years").

Paperless_Billing: Whether the customer opts for paperless billing (e.g., "Yes", "No").

Payment_Method: The payment method used by the customer (e.g., "Credit Card", "Bank Transfer", "Electronic Check").

Monthly_Charge: The amount the customer is charged monthly.

Total_Charges: The total amount of charges accumulated by the customer during their tenure.

Total_Refunds: The total refunds the customer has received.

Total_Extra_Data_Charges: The total charges for extra data usage.

Total_Long_Distance_Charges: The total charges for long-distance calls.

Total_Revenue: The total revenue generated from the customer.

Customer_Status: Whether the customer is active or has churned (e.g., "Active", "Churned").

Churn_Category: The category of churn (e.g., "Voluntary", "Involuntary").

Churn_Reason: The reason why the customer churned (e.g., "Better Offer", "Moved", etc.).

Data Usage Considerations:
Customer_Status is likely the target variable for your churn prediction model.

Gender, Age, State, Tenure_in_Months, Internet_Service, etc., are features that will be used in model training.

You might need to encode categorical variables such as Gender, State, Internet_Service, and others if using machine learning models like Decision Trees or Logistic Regression.
