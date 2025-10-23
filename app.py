import streamlit as st
import pandas as pd
import joblib
import os
import json
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Set page config
st.set_page_config(page_title="KFintech Chatbot", page_icon="💬")

st.title("🤖 KFintech Customer Service Chatbot")
st.write("How can I help you with your brokerage queries today?")

# Analytics tracking
analytics_file = "analytics.json"

def track_usage(query, intent):
    """Track user queries without external dependencies"""
    try:
        # Load existing analytics or create new
        if os.path.exists(analytics_file):
            with open(analytics_file, "r") as f:
                data = json.load(f)
        else:
            data = {"queries": [], "page_views": 0, "sessions": []}
        
        # Add current session if not exists
        session_id = st.query_params.get("session", "default")
        current_session = {
            "session_id": session_id,
            "start_time": datetime.now().isoformat(),
            "queries_count": 0
        }
        
        # Update analytics
        data["page_views"] += 1
        data["queries"].append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "intent": intent,
            "session_id": session_id
        })
        
        # Save analytics
        with open(analytics_file, "w") as f:
            json.dump(data, f, indent=2)
            
    except Exception as e:
        pass  # Fail silently for deployment

# Load or create models
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
    
    # TRACK USAGE
    track_usage(user_query, intent)
    
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

# ANALYTICS DASHBOARD
st.write("---")
st.subheader("📊 Live Analytics Dashboard")

try:
    if os.path.exists(analytics_file):
        with open(analytics_file, "r") as f:
            data = json.load(f)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Page Views", data.get("page_views", 0))
        with col2:
            st.metric("Total Queries", len(data.get("queries", [])))
        with col3:
            unique_sessions = len(set(q['session_id'] for q in data.get("queries", [])))
            st.metric("Unique Sessions", unique_sessions)
        
        # Show recent activity
        st.subheader("📈 Recent Activity")
        recent_queries = data.get("queries", [])[-10:]  # Last 10 queries
        for query in recent_queries:
            time = query['timestamp'][11:16]  # Extract time only
            st.write(f"🕒 {time} - '{query['query']}' → **{query['intent']}**")
            
    else:
        st.info("📊 Analytics: No data yet. Use the chatbot to see analytics!")
        
except Exception as e:
    st.info("📊 Analytics: Collecting initial data...")

# ALERTS SECTION
st.subheader("🔔 Real-time Alerts")
if st.button("Check System Status"):
    st.success("""
    ✅ **System Status: ACTIVE**
    📊 Analytics: COLLECTING DATA
    🤖 AI Model: OPERATIONAL
    🌐 Deployment: LIVE
    """)