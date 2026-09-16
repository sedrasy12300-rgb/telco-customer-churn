import streamlit as st
import joblib
import pandas as pd

st.markdown("""
<style>
[data-testid ="stAppViewContainer"]{
background-color :#D6E4E8;
}
[data-testid="stHeading"]{
font-size:42px; 
color:#234E70;}
h3{
font-size: 24px !important;
color : #234E70;}
.stButton button{
background-color:#234E70;
color :white;
font-size:18px;
border-radius:10px;
width :220px;
}
</style>
""",unsafe_allow_html= True)
st.title(" Customer Churn Prediction ")
st.write("Predict whe0ther  a customer is likely to leave the service ")
st.subheader("Customer Information")

col1,col2 =st.columns(2)
with col1:
    gender =st.selectbox("Gender",["Male","Female"])

with col2:
    Senior_Citizen=st.selectbox("Senior Citizen",["Yes","No"])

col3 ,col4 =st.columns(2)
with col3:
    Partner=st.selectbox("Partner",["Yes","No"])
with col4:
    Dependents=st.selectbox("Dependents",["Yes","No"])
Tenure=st.number_input("Tenure",min_value=0,max_value=72)

st.subheader("Service")

col5,col6,col7 =st.columns(3)
with col5:
    Phone_Service=st.selectbox("Phon Service",["Yes","No"])
with col6:
    Multiple_Lines=st.selectbox("Multiple Lines",["Yes","No","No phone service"])
with col7:
    Internet_Service=st.selectbox("Internet Service",["DSL","Fiber optic","No"])

col8,col9,col10 =st.columns(3)
with col8:
    Online_Security= st.selectbox("Online Security",["Yes","No","No internet service"])

with col9:
    Online_Backup=st.selectbox("Online Backup",["Yes","No","No internet service"])
with col10:
    Device_Protection=st.selectbox("Device Protection",["Yes","No","No internet service"])
col11,col12,col13 =st.columns(3)
with col11:

    Tech_Support= st.selectbox("Tech Support",["Yes","No","No internet service"])
with col12:
    Streaming_TV=st.selectbox("Streaming TV",["Yes","No","No internet service"])
with col13:
    Streaming_Movies=st.selectbox("Streaming Movies",["Yes","No","No internet service"])


st.subheader("Billing & Contract")
col14,col15,col16 =st.columns(3)
with col14:
    Contract=st.selectbox("Contract",["Month-to-month","One year","Two year"])
with col15:
    Paperless_Billing=st.selectbox("Paperless Billing",["Yes","No"])
with col16:
    Payment_Method=st.selectbox("Payment Method",["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])

col17,col18 =st.columns(2)
with col17:
    Monthly_Charges=st.number_input("Monthly Charges",min_value=0.0)

with col18:
    Total_Charges=st.number_input("Total Charges",min_value=0.0)


model =joblib.load("telco_churn_model.pkl")


if st.button("Predict Customer Churn"):
    st.subheader("Prediction Result")
    input_data=pd.DataFrame({
        "gender" :[gender],
        "SeniorCitizen":[Senior_Citizen],
        "Partner":[Partner],
        "Dependents":[Dependents],
        "tenure":[Tenure],
        "PhoneService":[Phone_Service],
        "MultipleLines":[Multiple_Lines],
        "InternetService":[Internet_Service],
        "OnlineSecurity":[Online_Security],
        "OnlineBackup":[Online_Backup],
        "DeviceProtection":[Device_Protection],
        "TechSupport":[Tech_Support],
        "StreamingTV":[Streaming_TV],
        "StreamingMovies":[Streaming_Movies],
        "Contract":[Contract],
        "PaperlessBilling":[Paperless_Billing],
        "PaymentMethod":[Payment_Method],
        "MonthlyCharges":[Monthly_Charges],
        "TotalCharges" :[Total_Charges],
    })
    input_data_columns=input_data.select_dtypes(include="object").columns
    input_data =pd.get_dummies(input_data,columns=input_data_columns)
    input_data =input_data.reindex(columns=model.feature_names_in_,fill_value=0)
    prediction =model.predict(input_data)
    churn_probability =model.predict_proba(input_data)[0][1]
    churn_probability =churn_probability * 100
    if prediction ==1:
        st.warning("Customer is likely to churn ")
        st.warning(f"Churn Probability : {churn_probability :.2f}%")
        
    else:
        st.success("Customer is likely to stay")
        st.success(f"Churn Probability : {churn_probability :.2f}%")
        
