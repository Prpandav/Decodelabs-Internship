import streamlit as st
import joblib
import os

# Set page configuration
st.set_page_config(page_title="AI News Classifier", page_icon="📰")

# 1. Load the saved model, vectorizer, and encoder
# Using @st.cache_resource ensures they are only loaded once, making the app much faster
@st.cache_resource
def load_components():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(base_dir, 'lr_news_model.pkl'))
    vectorizer = joblib.load(os.path.join(base_dir, 'tfidf_vectorizer.pkl'))
    encoder = joblib.load(os.path.join(base_dir, 'label_encoder.pkl'))
    return model, vectorizer, encoder

try:
    lr_model, tfidf_vectorizer, label_encoder = load_components()
except FileNotFoundError:
    st.error("Error: Could not find the required model files (.pkl). Please ensure they are in the same directory as this script.")
    st.stop()

# 2. Build the App UI
st.title("AI News Classifier")
st.write("Type or paste a news headline or snippet below, and our Logistic Regression model will predict its category!")

# Text area for user input
user_input = st.text_area("Enter News Content:", height=150, placeholder="e.g., The stock market closed at an all-time high today...")

# 3. Prediction Logic
if st.button("Predict Category", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        # Clean the text
        cleaned_text = user_input.lower()
        
        # Vectorize
        vectorized_text = tfidf_vectorizer.transform([cleaned_text])
        
        # Predict
        predicted_label = lr_model.predict(vectorized_text)
        predicted_category = label_encoder.inverse_transform(predicted_label)[0]
        
        # Display the Result
        st.success("Prediction Complete!")
        st.markdown(f"###  Predicted Category: **{predicted_category.title()}**")

# Footer
st.markdown("---")
st.markdown("Made by @Prpandav | Powered by Logistic Regression & TF-IDF Vectorization")