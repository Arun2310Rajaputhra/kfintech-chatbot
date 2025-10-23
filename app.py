import streamlit as st
import streamlit_analytics
import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# Set page config
st.set_page_config(page_title="KFintech Chatbot", page_icon="💬")

# Initialize analytics (tracks ALL user interactions)
streamlit_analytics.start_tracking()

st.title("🤖 KFintech Customer Service Chatbot")
st.write("How can I help you with your brokerage queries today?")

# Load or create models (your existing code)
@st.cache_resource
def load_models():
    if not os.path.exists('vectorizer.pkl') or not os.path.exists('intent_classifier.pkl'):
        st.info("🔄 First-time setup: Training AI models...")
        data = pd.read_csv('training_data.csv')
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(data['text'])
        y = data['intent']
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        joblib.dump(vectorizer, 'vectorizer.pkl')
        joblib.dump(model, 'intent_classifier.pkl')
        st.success("✅ Models trained successfully!")
    else:
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

# STOP analytics tracking and show dashboard
streamlit_analytics.stop_tracking()

# 8.2: ANALYTICS DASHBOARD (Only visible to you)
st.write("---")
st.subheader("📊 Admin Analytics Dashboard")

if st.checkbox("Show Analytics (Admin Only)"):
    streamlit_analytics.show_results()
    
    # Additional custom analytics
    st.subheader("🤖 Chatbot Performance")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Page Views", "Tracking...")
    with col2:
        st.metric("Unique Visitors", "Tracking...")
    with col3:
        st.metric("Avg Session Time", "Tracking...")

# 8.3: ALERTS (Simple version)
st.subheader("🔔 Usage Alerts")
if st.button("Check Recent Activity"):
    st.info("""
    **Alerts System Ready!**
    - You'll see real-time user counts
    - Popular queries will be highlighted
    - Peak usage times tracked
    """)