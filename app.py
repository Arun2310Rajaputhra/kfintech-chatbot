import streamlit as st
import pandas as pd
import joblib

# Set page config
st.set_page_config(page_title="KFintech Chatbot", page_icon="💬")

st.title("🤖 KFintech Customer Service Chatbot")
st.write("How can I help you with your brokerage queries today?")

# Load models (with caching)
@st.cache_resource
def load_models():
    vectorizer = joblib.load('vectorizer.pkl')
    model = joblib.load('intent_classifier.pkl')
    business_data = pd.read_csv('customer_database.csv')
    return vectorizer, model, business_data

vectorizer, model, business_data = load_models()

# Chat interface
user_query = st.text_input("Enter your query:", placeholder="e.g., Where is my withdrawal?")

if user_query:
    # NLP intent detection
    query_vec = vectorizer.transform([user_query])
    intent = model.predict(query_vec)[0]
    
    # Get business data
    business_info = business_data[business_data['query_type'] == intent.split('_')[0]]
    
    # Display results
    st.success(f"**Detected Intent:** {intent}")
    
    if not business_info.empty:
        details = business_info.iloc[0]['details']
        status = business_info.iloc[0]['status']
        processing_time = business_info.iloc[0]['processing_time']
        
        st.info(f"""
        **Details:** {details}
        **Status:** {status}
        **Processing Time:** {processing_time}
        """)
    
    st.write("💬 **Response:** I understand this is about", intent.replace('_', ' ').title(), 
             ". Our team will assist you shortly!")

st.write("---")
st.write("💡 **Try queries like:** 'where is my withdrawal', 'check investments', 'commission status'")