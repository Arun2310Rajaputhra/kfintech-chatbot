# STEP 5: LLM Approach with Prompt Engineering
import pandas as pd
import joblib
from transformers import pipeline

print("🧠 STEP 5: LLM Chatbot with Prompt Engineering")
print("=" * 50)

# 1. Load our trained model and data
print("1. Loading AI components...")
model = joblib.load('intent_classifier.pkl')
vectorizer = joblib.load('vectorizer.pkl')
business_data = pd.read_csv('customer_database.csv')

# 2. Load LLM for responses
print("2. Loading LLM for smart responses...")
chatbot = pipeline('text-generation', model='microsoft/DialoGPT-small')

# 3. Create response function
def get_llm_response(user_query, intent, business_info=""):
    """Use LLM to create human-like responses"""
    
    # Much simpler prompt
    prompt = f"Customer asks: '{user_query}'. Business info: {business_info}. As a helpful assistant, respond:"
    
    response = chatbot(prompt, max_new_tokens=80, num_return_sequences=1, temperature=0.8, do_sample=True)
    return response[0]['generated_text'].replace(prompt, "").strip()

# 4. Test the complete system
print("3. Testing complete system...")
test_queries = [
    "where is my withdrawal money?",
    "check my investment portfolio",
    "my commission is missing"
]

for query in test_queries:
    print(f"\n👤 User: {query}")
    
    # Step 1: NLP understands intent
    query_vec = vectorizer.transform([query])
    intent = model.predict(query_vec)[0]
    print(f"🤖 NLP Detected: {intent}")
    
    # Step 2: Get business data
    business_info = business_data[business_data['query_type'] == intent.split('_')[0]]
    
    # Step 3: LLM creates response
    if not business_info.empty:
        details = business_info.iloc[0]['details']
        status = business_info.iloc[0]['status']
        response = get_llm_response(query, intent, f"Status: {status}, Details: {details}")
    else:
        response = get_llm_response(query, intent)
    
    print(f"🧠 LLM Response: {response}")

print("\n4. ✅ LLM Chatbot system ready!")