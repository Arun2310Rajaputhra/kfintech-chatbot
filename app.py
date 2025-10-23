import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Set page config
st.set_page_config(page_title="KFintech Chatbot", page_icon="💬")

st.title("🤖 KFintech Customer Service Chatbot")
st.write("How can I help you with your brokerage queries today?")

# Load or create models
@st.cache_resource
def load_models():
    # Check if models exist, if not create them
    if not os.path.exists('vectorizer.pkl') or not os.path.exists('intent_classifier.pkl'):
        st.info("🔄 First-time setup: Training AI models...")
        
        # Load training data
        data = pd.read_csv('training_data.csv')
        
        # Create and train vectorizer
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(data['text'])
        y = data['intent']
        
        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        # Save models
        joblib.dump(vectorizer, 'vectorizer.pkl')
        joblib.dump(model, 'intent_classifier.pkl')
        
        st.success("✅ Models trained successfully!")
    else:
        # Load existing models
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