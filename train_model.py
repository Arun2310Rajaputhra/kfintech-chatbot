# STEP 4: Build Traditional ML Model
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

print("🤖 STEP 4: Building Machine Learning Model")
print("=" * 50)

# 1. Load our prepared features
print("1. Loading features and labels...")
vectorizer = joblib.load('vectorizer.pkl')
data = pd.read_csv('training_data.csv')

X = vectorizer.transform(data['text'])
y = data['intent']

# 2. Train the model
print("2. Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 3. Test the model
print("3. Testing model accuracy...")
predictions = model.predict(X)
accuracy = accuracy_score(y, predictions)
print(f"   Model Accuracy: {accuracy:.2%}")

# 4. Save the trained model
joblib.dump(model, 'intent_classifier.pkl')
print("4. ✅ Model saved as 'intent_classifier.pkl'")

# 5. Test with new examples
print("\n5. Testing with new queries:")
test_queries = [
    "where is my money withdrawal",
    "check my investment portfolio", 
    "commission not received"
]

for query in test_queries:
    query_vec = vectorizer.transform([query])
    prediction = model.predict(query_vec)[0]
    print(f"   '{query}' → {prediction}")
