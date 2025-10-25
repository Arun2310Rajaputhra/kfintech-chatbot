import streamlit as st
import pandas as pd
import joblib
import os
import hashlib
import psycopg2
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Set page config
st.set_page_config(page_title="KFintech Chatbot", page_icon="💬")

# ========================
# DATABASE CONNECTION
# ========================

def get_db_connection():
    """Create database connection using Streamlit secrets"""
    try:
        conn = psycopg2.connect(
            host=st.secrets["postgres"]["host"],
            port=st.secrets["postgres"]["port"],
            database=st.secrets["postgres"]["database"],
            user=st.secrets["postgres"]["user"],
            password=st.secrets["postgres"]["password"],
            sslmode="require"
        )
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

def init_app_tables():
    """Initialize only app-specific tables (users, analytics)"""
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            
            # Create users table (for authentication)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    role VARCHAR(20) DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create user_queries table (for analytics)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_queries (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(50) NOT NULL,
                    query_text TEXT NOT NULL,
                    intent VARCHAR(100) NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert default users
            cur.execute("""
                INSERT INTO users (username, password_hash, role) 
                VALUES 
                ('admin', '5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8', 'admin'),
                ('user', '5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8', 'user')
                ON CONFLICT (username) DO NOTHING
            """)
            
            conn.commit()
            cur.close()
            conn.close()
            
        except Exception as e:
            st.error(f"Database setup failed: {e}")

# ========================
# DATABASE OPERATIONS (READ/WRITE ONLY)
# ========================

def get_business_data(query_type):
    """READ business data from Neon database"""
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT customer_id, details, status, processing_time FROM business_data WHERE query_type = %s",
                (query_type,)
            )
            result = cur.fetchone()
            cur.close()
            conn.close()
            return result
        except Exception as e:
            st.error(f"Error reading business data: {e}")
    return None

def track_usage(query, intent, username):
    """WRITE user query to Neon database"""
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO user_queries (username, query_text, intent) VALUES (%s, %s, %s)",
                (username, query, intent)
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            st.error(f"Error tracking usage: {e}")

def get_analytics():
    """READ analytics from Neon database"""
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            
            # Total queries
            cur.execute("SELECT COUNT(*) FROM user_queries")
            total_queries = cur.fetchone()[0]
            
            # Unique users
            cur.execute("SELECT COUNT(DISTINCT username) FROM user_queries")
            unique_users = cur.fetchone()[0]
            
            # Recent queries
            cur.execute("""
                SELECT username, query_text, intent, timestamp 
                FROM user_queries 
                ORDER BY timestamp DESC 
                LIMIT 10
            """)
            recent_queries = cur.fetchall()
            
            cur.close()
            conn.close()
            
            return {
                "total_queries": total_queries,
                "unique_users": unique_users,
                "recent_queries": recent_queries
            }
        except Exception as e:
            st.error(f"Error reading analytics: {e}")
    return None

# ========================
# AUTHENTICATION (Same as before)
# ========================

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(username, password):
    conn = get_db_connection()
    if conn:
        try:
            cur = conn.cursor()
            hashed_input = hash_password(password)
            cur.execute(
                "SELECT username, role FROM users WHERE username = %s AND password_hash = %s",
                (username, hashed_input)
            )
            result = cur.fetchone()
            cur.close()
            conn.close()
            if result:
                return {"username": result[0], "role": result[1]}
        except Exception as e:
            st.error(f"Authentication error: {e}")
    return None

def init_session_state():
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
    
    # Initialize app tables on first load
    if st.button("Initialize App"):
        init_app_tables()
        st.success("App tables initialized!")
    
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        submit = st.form_submit_button("Login")
        
        if submit:
            user_info = authenticate_user(username, password)
            if user_info:
                st.session_state.authenticated = True
                st.session_state.user_role = user_info["role"]
                st.session_state.username = user_info["username"]
                st.success(f"✅ Welcome, {user_info['username']}!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")
    
    st.info("""
    **Demo Credentials:**
    - **Admin:** username: `admin`, password: `password`  
    - **User:** username: `user`, password: `password`
    """)
    
    st.stop()

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

# Load ML models
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
    return vectorizer, model

vectorizer, model = load_models()

# ========================
# CHATBOT INTERFACE
# ========================

st.write("How can I help you with your brokerage queries today?")

user_query = st.text_input("Enter your query:", placeholder="e.g., Where is my withdrawal?")

if user_query:
    # NLP intent detection
    query_vec = vectorizer.transform([user_query])
    intent = model.predict(query_vec)[0]
    
    # TRACK USAGE in database (WRITE)
    track_usage(user_query, intent, st.session_state.username)
    
    # Get business data from database (READ)
    business_info = get_business_data(intent.split('_')[0])
    
    # Display results
    st.success(f"**Detected Intent:** {intent}")
    
    if business_info:
        customer_id, details, status, processing_time = business_info
        st.info(f"""
        **Customer ID:** {customer_id}
        **Details:** {details}
        **Status:** {status}
        **Processing Time:** {processing_time}
        """)
    
    st.write("💬 **Response:** I understand this is about", intent.replace('_', ' ').title(), 
             ". Our team will assist you shortly!")

st.write("---")
st.write("💡 **Try queries like:** 'where is my withdrawal', 'check investments', 'commission status'")

# ========================
# ADMIN DASHBOARD
# ========================

if st.session_state.user_role == "admin":
    st.write("---")
    st.subheader("👑 Admin Dashboard")
    
    # Database Analytics (READ)
    st.subheader("📊 Database Analytics")
    
    analytics_data = get_analytics()
    if analytics_data:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Queries", analytics_data["total_queries"])
        with col2:
            st.metric("Unique Users", analytics_data["unique_users"])
        with col3:
            st.metric("Data Source", "Neon PostgreSQL")
        
        # Recent queries from database
        st.write("**Recent Queries:**")
        for query in analytics_data["recent_queries"]:
            username, query_text, intent, timestamp = query
            time = timestamp.strftime("%H:%M") if hasattr(timestamp, 'strftime') else str(timestamp)[11:16]
            st.write(f"🕒 {time} - **{username}**: '{query_text}' → **{intent}**")

# System Status
st.write("---")
st.subheader("🔔 System Status")
st.success(f"✅ Logged in as: {st.session_state.username}")
st.info(f"🗄️ Database: Neon PostgreSQL | 🤖 AI: Operational | 📊 Data: Real-time")