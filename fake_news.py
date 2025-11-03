import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# ---- Load the datasets ----
fake = pd.read_csv('data/Fake.csv')
true = pd.read_csv('data/True.csv')

# ---- Add labels ----
fake['label'] = 1   # 1 = Fake
true['label'] = 0   # 0 = True/Real

# ---- Combine the datasets ----
df = pd.concat([fake, true], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

# ---- Prepare data ----
df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
X = df['content']
y = df['label']

# ---- Split data ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- Vectorize text ----
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ---- Train model ----
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# ---- Evaluate ----
y_pred = model.predict(X_test_tfidf)
print("✅ Model Training Completed!")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---- Save model & vectorizer ----
os.makedirs('models', exist_ok=True)
with open('models/fake_news_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("\n✅ Model and vectorizer saved in 'models/' folder!")
