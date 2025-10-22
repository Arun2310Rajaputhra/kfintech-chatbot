# STEP 3: Explore and Prepare Data
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

print("📊 STEP 3: Data Exploration & Feature Engineering")
print("=" * 50)

# 1. Load data
data = pd.read_csv('training_data.csv')
print("1. Data loaded:", data.shape)

# 2. Show unique intents
print("\n2. Customer query types (intents):")
print(data['intent'].value_counts())

# 3. Feature Engineering - Convert text to numbers
print("\n3. Converting text to numbers (Feature Engineering)...")
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(data['text'])
y = data['intent']

print("   Text features shape:", X.shape)
print("   Feature names sample:", list(vectorizer.get_feature_names_out())[:10])

# 4. Split data for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\n4. Data split for training:")
print("   Training samples:", X_train.shape[0])
print("   Testing samples:", X_test.shape[0])

# 5. Save the vectorizer for later use
import joblib
joblib.dump(vectorizer, 'vectorizer.pkl')
print("\n5. ✅ Feature engineering completed!")
print("   Vectorizer saved as 'vectorizer.pkl'")