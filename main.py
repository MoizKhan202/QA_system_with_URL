import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')  # Download punkt tokenizer if not already present

@st.cache_resource
def load_models():
    """Loads the tokenizer and QA model. Cached for efficiency."""
    tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
    model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    return qa_pipeline

def extract_text_from_url(url):
    """Extracts text content from a given URL."""
    try:
        response = requests.get(url, timeout=10) # Added timeout
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text(separator="\n")
        #Improved text cleaning
        text = " ".join(text.split()) #remove extra whitespaces and newlines
        return text
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching URL: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

def split_into_chunks(text, chunk_size=500):
    """Splits text into smaller chunks."""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
      if len(current_chunk) + len(sentence) + 1 <= chunk_size:
        current_chunk += sentence + " "
      else:
        chunks.append(current_chunk.strip())
        current_chunk = sentence + " "
    chunks.append(current_chunk.strip()) # Add last chunk
    return chunks

# Streamlit app
st.title("Webpage Question Answering")

url = st.text_input("Enter URL:", placeholder="e.g., https://en.wikipedia.org/wiki/Main_Page")
question = st.text_input("Enter your question:")

qa_pipeline = load_models()

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
              try: #Added try-except block for QA
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