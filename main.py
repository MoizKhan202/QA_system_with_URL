import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
import torch
import os

# Set NLTK data path
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

# Download punkt only if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)

@st.cache_resource
def load_models():
    try:
        tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
        model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
        qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
        return qa_pipeline
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

def extract_text_from_url(url):
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text(separator="\n")
        text = " ".join(text.split())
        return text
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching URL: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

def split_into_chunks(text, chunk_size=500):
    
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    chunks.append(current_chunk.strip())
    return chunks

# Streamlit app
st.title("Webpage Question Answering")

url = st.text_input("Enter URL:", placeholder="e.g., https://en.wikipedia.org/wiki/Main_Page")
question = st.text_input("Enter your question:")

qa_pipeline = load_models()

if qa_pipeline is None:
    st.stop()

if st.button("Submit"):
    if not url:
        st.warning("Please enter a URL.")
    elif not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Fetching and processing webpage..."):
            context = extract_text_from_url(url)
            if context:
                chunks = split_into_chunks(context)
                answers = []
                for i, chunk in enumerate(chunks):
                    try:
                        result = qa_pipeline(question=question, context=chunk)
                        answers.append({"chunk_index": i, "answer": result['answer'], "score": result['score']})
                    except Exception as e:
                        st.error(f"Error during QA on chunk {i+1}: {e}")

                if answers:
                    best_answer = max(answers, key=lambda x: x['score'])
                    st.write("**Answer:**", best_answer['answer'])
                    st.write("**Score:**", best_answer['score'])
                else:
                    st.write("Could not find an answer in the provided context.")
            else:
                st.write("Could not retrieve content from the provided URL.")
