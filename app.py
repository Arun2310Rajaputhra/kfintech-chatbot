import streamlit as st
import pandas as pd
import joblib
import os
import json
import hashlib
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Set page config
st.set_page_config(page_title="KFintech Chatbot", page_icon="💬")

# ========================
# AUTHENTICATION SYSTEM
# ========================

# User database (in production, use real database)
USER_DB = {
    "admin": {
        "password": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",  # "password"
        "role": "admin",
        "name": "Administrator"
    },
    "user": {
        "password": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",  # "password"  
        "role": "user",
        "name": "Regular User"
    }
}

def hash_password(password):
    """Hash password for security"""
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(username, password):
    """Authenticate user credentials"""
    if username in USER_DB:
        hashed_input = hash_password(password)
        if USER_DB[username]["password"] == hashed_input:
            return USER_DB[username]
    return None

def init_session_state():
    """Initialize session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    if 'username' not in st.session_state:
        st.session_state.username = None

init_session_state()

# ========================
# LOGIN PAGE
# ========================

if not st.session_state.authenticated:
    st.title("🔐 KFintech Login")
    st.write("Please login to access the AI Chatbot")
    
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        submit = st.form_submit_button("Login")
        
        if submit:
            user_info = authenticate_user(username, password)
            if user_info:
                st.session_state.authenticated = True
                st.session_state.user_role = user_info["role"]
                st.session_state.username = user_info["name"]
                st.success(f"✅ Welcome, {user_info['name']}!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")
    
    # Demo credentials
    st.info("""
    **Demo Credentials:**
    - **Admin:** username: `admin`, password: `password`  
    - **User:** username: `user`, password: `password`
    """)
    
    st.stop()  # Stop here if not authenticated

# ========================
# MAIN APPLICATION
# ========================

st.title(f"🤖 KFintech Customer Service Chatbot")
st.write(f"Welcome, **{st.session_state.username}**! | Role: **{st.session_state.user_role.title()}**")

# Logout button
if st.sidebar.button("🚪 Logout"):
    st.session_state.authenticated = False
    st.session_state.user_role = None
    st.session_state.username = None
    st.rerun()

# Analytics tracking (only for authenticated users)
analytics_file = "analytics.json"

def track_usage(query, intent, username):
    """Track user queries"""
    try:
        if os.path.exists(analytics_file):
            with open(analytics_file, "r") as f:
                data = json.load(f)
        else:
            data = {"queries": [], "page_views": 0, "user_activity": {}}
        
        # Update analytics
        data["page_views"] += 1
        
        # Track user-specific activity
        if username not in data["user_activity"]:
            data["user_activity"][username] = {"query_count": 0, "last_active": ""}
        
        data["user_activity"][username]["query_count"] += 1
        data["user_activity"][username]["last_active"] = datetime.now().isoformat()
        
        data["queries"].append({
            "timestamp": datetime.now().isoformat(),
            "username": username,
            "role": st.session_state.user_role,
            "query": query,
            "intent": intent
        })
        
        with open(analytics_file, "w") as f:
            json.dump(data, f, indent=2)
            
    except Exception as e:
        pass

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

# ========================
# CHATBOT INTERFACE
# ========================

st.write("How can I help you with your brokerage queries today?")

user_query = st.text_input("Enter your query:", placeholder="e.g., Where is my withdrawal?")

if user_query:
    # NLP intent detection
    query_vec = vectorizer.transform([user_query])
    intent = model.predict(query_vec)[0]
    
    # TRACK USAGE with username
    track_usage(user_query, intent, st.session_state.username)
    
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

# ========================
# ROLE-BASED FEATURES
# ========================

# ADMIN-ONLY FEATURES
if st.session_state.user_role == "admin":
    st.write("---")
    st.subheader("👑 Admin Dashboard")
    
    # User management
    st.write("**User Management**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("View All Users"):
            st.write("**Registered Users:**")
            for username, info in USER_DB.items():
                st.write(f"- {username} ({info['role']})")
    with col2:
        if st.button("System Status"):
            st.success("✅ All systems operational")
    
    # Analytics Dashboard (Admin only)
    st.subheader("📊 Advanced Analytics")
    
    try:
        if os.path.exists(analytics_file):
            with open(analytics_file, "r") as f:
                data = json.load(f)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Page Views", data.get("page_views", 0))
            with col2:
                st.metric("Total Queries", len(data.get("queries", [])))
            with col3:
                unique_users = len(data.get("user_activity", {}))
                st.metric("Active Users", unique_users)
            with col4:
                admin_queries = len([q for q in data.get("queries", []) if q.get('role') == 'admin'])
                st.metric("Admin Queries", admin_queries)
            
            # User activity breakdown
            st.write("**User Activity:**")
            for user, activity in data.get("user_activity", {}).items():
                st.write(f"- {user}: {activity['query_count']} queries")
                
            # Recent activity with usernames
            st.write("**Recent Activity:**")
            recent_queries = data.get("queries", [])[-10:]
            for query in recent_queries:
                time = query['timestamp'][11:16]
                st.write(f"🕒 {time} - **{query['username']}** asked '{query['query']}'")
                
        else:
            st.info("📊 Analytics: No data yet. Use the chatbot to see analytics!")
            
    except Exception as e:
        st.info("📊 Analytics: Loading...")

# REGULAR USER FEATURES
else:
    st.write("---")
    st.info("ℹ️ **User Tip:** Contact admin for analytics access and advanced features.")

# System Alerts
st.write("---")
st.subheader("🔔 System Status")
st.success(f"✅ Logged in as: {st.session_state.username} | Role: {st.session_state.user_role}")