import streamlit as st
import pickle

# Load model & vectorizer
model = pickle.load(open('../models/fake_news_model.pkl', 'rb'))
vectorizer = pickle.load(open('../models/vectorizer.pkl', 'rb'))

st.title("📰 Fake News Detection App")
st.write("Enter a news article below to check if it's real or fake.")

user_input = st.text_area("News content:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        data = vectorizer.transform([user_input])
        prediction = model.predict(data)[0]
        if prediction == 1:
            st.error("🚨 This news seems **Fake**.")
        else:
            st.success("✅ This news seems **Real**.")
