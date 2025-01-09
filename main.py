import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from bs4 import BeautifulSoup
import requests

# Initialize the QA model and tokenizer
st.title("Text Extractor and QA System 📖")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")

# Sidebar for URL input
st.sidebar.title("URL Input")
url = st.sidebar.text_input("Enter a URL to extract text:")

# Extract text from the URL
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract visible text
        for script in soup(["script", "style"]):
            script.extract()  # Remove JavaScript and CSS

        text = soup.get_text(separator="\n").strip()
        return text
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return None

# Display extracted text
if url:
    st.info("Extracting text from the provided URL...")
    context = extract_text_from_url(url)
    if context:
        st.success("Text extracted successfully!")
        st.text_area("Extracted Context", context[:1000], height=300)

# Tokenization
if context:
    tokens = tokenizer.tokenize(context)
    st.subheader("Tokenized Text")
    st.text_area("Tokenized Context", " ".join(tokens[:100]), height=200)

# Question Answering
st.subheader("Ask Questions")
question = st.text_input("Enter your question:")
if question and context:
    try:
        # Perform QA
        result = qa_pipeline(question=question, context=context)
        st.header("Answer")
        st.write(result["answer"])
    except Exception as e:
        st.error(f"Error processing the question: {e}")
