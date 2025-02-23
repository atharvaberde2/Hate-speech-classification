import streamlit as st
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification


st.markdown(
    """
    <style>
    .stApp {
        background-color: #1D1E22;
        color: white;
    }
    .stButton>button {
        background-color: #E63946;
        color: white;
        border-radius: 8px;
        font-size: 16px;
    }
    .stTextInput>div>div>input {
        background-color: #2C2F33;
        color: white;
    }
    .stMarkdown {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.title("🛑 Hate Speech Detector")

# User Input
text_input = st.text_area("Enter text:", placeholder="Type here...")
tokenizer1 = BertTokenizer.from_pretrained("Atharva1244/hate_speech_model")
model1 = TFBertForSequenceClassification.from_pretrained('Atharva1244/hate_speech_model')

# Submit Button
if st.button('Predict'):
    if text_input:
        # Tokenize the input text
        inputs = tokenizer1(text_input, return_tensors="tf", padding=True, truncation=True, max_length=512)

        # Get model predictions
        outputs = model1(**inputs)

        # Extract logits and convert to probabilities
        logits = outputs.logits
        probabilities = tf.nn.softmax(logits, axis=-1)

        # Get the predicted class
        predicted_class = tf.argmax(probabilities, axis=-1).numpy()

        # Display the result
        st.write(f"Predicted Class: {predicted_class[0]}")
        st.write(f"Probabilities: {probabilities.numpy()}")
    else:
        st.warning("Please enter some text to analyze.")
