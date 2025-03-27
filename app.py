import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from googletrans import Translator  # Import Translator

# Ensure required NLTK downloads
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Load the trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Initialize tools
ps = PorterStemmer()
translator = Translator()


# Function to preprocess text
def transform_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    # Filter only alphanumeric tokens that are not in stopwords
    filtered_words = [ps.stem(word) for word in words if word.isalnum() and word not in stopwords.words('english')]
    return " ".join(filtered_words)

# Streamlit UI
st.title("📩 SMS Spam Detection Model")
st.write("🔍 **Detect if a message is Spam or Not!**")
st.write("👨🏻‍💻 *Made with ❤️‍🔥 by Sadhika*")

# Input text box
input_sms = st.text_area("📩 Enter the SMS message here:", height=100)

# Predict button
if st.button("🔮 Predict"):

    # 1️ Detect Language
    detected_lang = translator.detect(input_sms).lang  # Detect language
    st.write(f"🌍 **Detected Language:** {detected_lang.upper()}")

    # 2️ Translate if not English
    if detected_lang != "en":
        translated_text = translator.translate(input_sms, dest="en").text  # Translate to English
        st.write(f"📝 **Translated Message:** {translated_text}")
    else:
        translated_text = input_sms  # No translation needed

    # 3️ Preprocess input
    transformed_sms = transform_text(translated_text)
    
    # 4️ Convert to TF-IDF vector
    vector_input = vectorizer.transform([transformed_sms])
    
    # 5️ Predict
    result = model.predict(vector_input)[0]

    # 6️ Display the result
    if result == 1:
        st.error("🚨 **Spam!** This message is likely spam.")
    else:
        st.success("✅ **Not Spam!** This message looks safe.")


