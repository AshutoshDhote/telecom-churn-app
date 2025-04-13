import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model and preprocessor
model = joblib.load("models/churn_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

# Load data for dropdown options
df = pd.read_csv("data/Customer_Data.csv")
df = df[df['Customer_Status'] != 'Joined']  # Filter to match training set

# App Layout
st.set_page_config(page_title="Telecom Churn Predictor", layout="wide")
st.title("📉 Telecom Customer Churn Analysis")

st.markdown("""
    <style>
    /* Tab styling */
    [data-baseweb="tab"] {
        font-size: 18px;
        font-weight: 600;
        padding: 12px 24px;
        border-radius: 6px 6px 0 0;
        transition: transform 0.2s ease-in-out;
    }

    /* Zoom effect on hover */
    [data-baseweb="tab"]:hover {
        transform: scale(1.08);
        background-color: #e6f0ff;
        cursor: pointer;
    }

    /* Selected tab style */
    [data-baseweb="tab"][aria-selected="true"] {
        background-color: #f0f2f6;
        border-bottom: 3px solid #0e1117;
        color: #000000;
    }

    /* Spacing between tabs */
    [data-baseweb="tab-list"] {
        gap: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Define Tabs
tabs = st.tabs(["🧠 **Predict Churn**", "⚖️ Most Likely to Churn/Stay", "📈 Power BI Report", "📌 Key Metrics", "📋 Sample Inputs"])

# ----------------------------- TAB 1: PREDICT CHURN -----------------------------
with tabs[0]:
    st.header("🧠 Predict Customer Churn")

    # ----------------------------- Customer Info Section -----------------------------
    st.markdown("---")
    st.markdown("### 🧍 Customer Info")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", sorted(df['Gender'].dropna().unique()))
        married = st.selectbox("Married", sorted(df['Married'].dropna().unique()))
        state = st.selectbox("State", sorted(df['State'].dropna().unique()))
    with col2:
        age = st.slider("Age", int(df['Age'].min()), int(df['Age'].max()), int(df['Age'].median()))
        number_of_referrals = st.slider("Number of Referrals", 
                                        int(df['Number_of_Referrals'].min()), 
                                        int(df['Number_of_Referrals'].max()), 
                                        int(df['Number_of_Referrals'].median()))
        tenure_in_months = st.slider("Tenure (Months)", 
                                    int(df['Tenure_in_Months'].min()), 
                                    int(df['Tenure_in_Months'].max()), 
                                    int(df['Tenure_in_Months'].median()))

    # -------------------------- Service & Internet Logic  --------------------------
    st.markdown("---")
    st.markdown("### 🛠️ Service Add-Ons")

    # Service add-ons options (Phone Service, Multiple Lines)
    col3, col4 = st.columns(2)
    with col3:
        #Internet Service Header
        st.subheader("Internet Service")

        #Choose if the customer has Internet Service
        internet_service = st.selectbox("Choose if the customer has Internet Service", ["Yes", "No"])
        
        # Initialize variables with empty string as default values
        online_security = ""
        online_backup = ""
        internet_type = ""
        premium_support = ""
        device_protection_plan = ""

        # Show related service options only if Internet Service is "Yes"
        if internet_service == "Yes":
            online_security = st.selectbox("Online Security", sorted(df['Online_Security'].dropna().unique()))
            online_backup = st.selectbox("Online Backup", sorted(df['Online_Backup'].dropna().unique()))
            internet_type = st.selectbox("Internet Type", sorted(df['Internet_Type'].dropna().unique()))
            premium_support = st.selectbox("Premium Support", sorted(df['Premium_Support'].dropna().unique()))
            device_protection_plan = st.selectbox("Device Protection Plan", sorted(df['Device_Protection_Plan'].dropna().unique()))
        
        # Don't show related service options as Internet Service is "No"
        else:
            # Grey out the fields by using `disabled=True`
            online_security = st.selectbox("Online Security", sorted(df['Online_Security'].dropna().unique()), disabled=True)
            online_backup = st.selectbox("Online Backup", sorted(df['Online_Backup'].dropna().unique()), disabled=True)
            internet_type = st.selectbox("Internet Type", ["N/A"], disabled=True)
            premium_support = st.selectbox("Premium Support", sorted(df['Premium_Support'].dropna().unique()), disabled=True)
            device_protection_plan = st.selectbox("Device Protection Plan", sorted(df['Device_Protection_Plan'].dropna().unique()), disabled=True)

    #Phone Service Header
    with col4:
        st.subheader("Phone Service")
        phone_service = st.selectbox("Phone Service", sorted(df['Phone_Service'].dropna().unique()))
        multiple_lines = st.selectbox("Multiple Lines", sorted(df['Multiple_Lines'].dropna().unique()))

    # ----------------------------- Billing Info Section -----------------------------
    st.markdown("---")
    st.markdown("### 💳 Billing Info")
    col5, col6 = st.columns(2)

    with col5:
        contract = st.selectbox("Contract", sorted(df['Contract'].dropna().unique()))
        paperless_billing = st.selectbox("Paperless Billing", sorted(df['Paperless_Billing'].dropna().unique()))
        payment_method = st.selectbox("Payment Method", sorted(df['Payment_Method'].dropna().unique()))
        monthly_charge = st.number_input("Monthly Charge (INR)", 
                                        min_value=float(df['Monthly_Charge'].min()), 
                                        max_value=float(df['Monthly_Charge'].max()), 
                                        value=float(df['Monthly_Charge'].median()))

        total_charges = st.number_input("Total Charges (INR)", 
                                        min_value=float(df['Total_Charges'].min()), 
                                        max_value=float(df['Total_Charges'].max()), 
                                        value=float(df['Total_Charges'].median()))
    with col6:
        total_extra_data_charges = st.number_input("Total Extra Data Charges", min_value=0.0, value=0.0)
        total_refunds = st.number_input("Total Refunds", min_value=0.0, value=0.0)
        total_long_distance_charges = st.number_input("Total Long Distance Charges", min_value=0.0, value=0.0)
        unlimited_data = st.selectbox("Unlimited Data", ['Yes', 'No'])
        total_revenue = st.number_input("Total Revenue", min_value=0.0, value=700.0)

    # ----------------------------- Submit Button -----------------------------
    st.markdown("<br>", unsafe_allow_html=True) 
    submit = st.button("🔍 Predict Churn")

    # On submit
    if submit:
        input_data = pd.DataFrame([{
            "Gender": gender,
            "Age": age,
            "Married": married,
            "State": state,
            "Number_of_Referrals": number_of_referrals,
            "Tenure_in_Months": tenure_in_months,
            "Phone_Service": phone_service,
            "Multiple_Lines": multiple_lines,
            "Internet_Service": internet_service,
            "Internet_Type": internet_type,
            "Online_Security": online_security,
            "Online_Backup": online_backup,
            "Device_Protection_Plan": device_protection_plan,
            "Premium_Support": premium_support,
            "Contract": contract,
            "Paperless_Billing": paperless_billing,
            "Payment_Method": payment_method,
            "Monthly_Charge": monthly_charge,
            "Total_Charges": total_charges,
            "Total_Extra_Data_Charges": total_extra_data_charges,
            "Total_Refunds": total_refunds,
            "Total_Long_Distance_Charges": total_long_distance_charges,
            "Unlimited_Data": unlimited_data,
            "Total_Revenue": total_revenue
        }])

        # Preprocess and predict
        transformed = preprocessor.transform(input_data)
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        st.subheader("📊 Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ The customer is likely to churn. (Confidence: {prob:.2%})")
        else:
            st.success(f"✅ The customer is likely to stay. (Confidence: {1 - prob:.2%})")
# ----------------------------- TAB 2: LIKELY TO CHURN OR STAY -----------------------------
with tabs[1]:
    st.header("⚖️ Most Likely to Churn / Stay")

    # Get churn probabilities
    churn_probs = model.predict_proba(df)[:, 1]
    df_extreme = df.copy()
    df_extreme['Churn_Probability'] = churn_probs

    # Get top 3 churners and top 3 stayers
    top_3_churners = df_extreme.sort_values('Churn_Probability', ascending=False).head(3)
    top_3_stayers = df_extreme.sort_values('Churn_Probability').head(3)

    st.subheader("🚨 Top 3 Customers Most Likely to Churn")
    st.dataframe(top_3_churners, use_container_width=True)
    for _, row in top_3_churners.iterrows():
        customer_id = row['Customer_ID']  # Replace with the correct column name if it's different
        st.markdown(f"**Customer ID: {customer_id} - Confidence (to churn): {row['Churn_Probability']:.2%}**")

    st.subheader("💚 Top 3 Customers Most Likely to Stay")
    st.dataframe(top_3_stayers, use_container_width=True)
    for _, row in top_3_stayers.iterrows():
        customer_id = row['Customer_ID']  # Replace with the correct column name if it's different
        st.markdown(f"**Customer ID: {customer_id} - Confidence (to stay): {1 - row['Churn_Probability']:.2%}**")

# ----------------------------- TAB 3: POWER BI REPORT -----------------------------
with tabs[2]:
    st.header("📈 Power BI Report")
    st.markdown("Below is a sample embedded Power BI report. Replace the iframe `src` with your actual report link.")
    
    power_bi_html = """
    <iframe title="PowerBI Report" width="100%" height="600"
        src="https://app.powerbi.com/view?r=eyJrIjoiZjQ1MDlmNzEtNjY2Ny00MGE3LTk0ODEtODBkM2U3YzUyYmU1IiwidCI6IjE0MjQzYzM5LTQwNGMtNDI0Yi05ZTZjLTY0MDUwNmM3YmFlZiJ9"
        frameborder="0" allowFullScreen="true"></iframe>
    """
    st.components.v1.html(power_bi_html, height=620)

# ----------------------------- TAB 4: KEY METRICS -----------------------------
with tabs[3]:
    st.header("📌 Key Metrics")

    churn_rate = df[df['Customer_Status'] == 'Churned'].shape[0] / df.shape[0]
    avg_monthly = df['Monthly_Charge'].mean()
    avg_tenure = df['Tenure_in_Months'].mean()
    churn_by_state = df[df['Customer_Status'] == 'Churned']['State'].value_counts().head(5)

    st.metric("🔻 Churn Rate", f"{churn_rate:.2%}")
    st.metric("💵 Avg Monthly Charge", f"INR {avg_monthly:.2f}")
    st.metric("📅 Avg Tenure", f"{avg_tenure:.1f} months")

    st.subheader("🌍 Top 5 States by Churn Count")
    st.bar_chart(churn_by_state)

# ----------------------------- TAB 5: SAMPLE INPUTS -----------------------------
with tabs[4]:
    st.header("📋 Sample Input Data (for testing)")
    st.write("Use these example rows as a reference for the form in the **Predict Churn** tab.")
    st.dataframe(df.sample(5).reset_index(drop=True))
