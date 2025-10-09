# Intelligent Deep Learning Chatbot using PDF Data + API + Linear Regression

import os
import re
import PyPDF2
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

nltk.download('stopwords')

# ---------- Step 1: Read PDF Files ----------
def extract_text_from_pdfs(folder_path):
    all_text = ""
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)
            with open(pdf_path, 'rb') as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                for page in reader.pages:
                    all_text += page.extract_text()
    return all_text

# ---------- Step 2: Preprocess Text ----------
def preprocess_text(text):
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.]', ' ', text)
    words = text.lower().split()
    stop_words = set(stopwords.words('english'))
    # Keep more words, only remove stop words if there are enough remaining words
    clean_words = [w for w in words if w.isalnum() and len(w) > 2]
    # If removing stop words leaves us with very few words, keep them
    if len([w for w in clean_words if w not in stop_words]) < 10:
        return " ".join(clean_words)
    else:
        clean_words = [w for w in clean_words if w not in stop_words]
        return " ".join(clean_words)

# ---------- Step 3: Prepare Data ----------
pdf_folder = "pdf data"  # Folder containing your PDFs
raw_text = extract_text_from_pdfs(pdf_folder)
clean_text = preprocess_text(raw_text)

# Split into chunks (like Q&A pairs)
sentences = [s.strip() for s in clean_text.split('.') if len(s.strip()) > 20]
print(f"Found {len(sentences)} sentences from PDFs")

# Check if we have enough content
if len(sentences) < 5:
    print("Warning: Very few sentences found. Check your PDF content.")
    # Add some default sentences but keep the PDF content
    sentences.extend(["I can help you with questions about the documents.", 
                     "Please ask me something specific about the content."])

vectorizer = TfidfVectorizer(min_df=1, max_features=5000, stop_words='english')
X = vectorizer.fit_transform(sentences)
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"Processing {len(sentences)} text chunks")

# ---------- Step 4: Simple Similarity-Based Search (More Effective) ----------

# ---------- Step 5: Chatbot Function ----------
def chatbot_response(user_input):
    try:
        # Transform user input to same vector space
        user_vec = vectorizer.transform([user_input])
        
        if user_vec.sum() == 0:
            return "Bot: I'm sorry, I couldn't find relevant information about that topic in the documents."
        
        # Calculate similarity between user input and all sentences
        similarities = cosine_similarity(user_vec, X).flatten()
        
        # Get top 3 most similar sentences
        top_indices = similarities.argsort()[-3:][::-1]
        top_similarities = similarities[top_indices]
        
        # Filter out very low similarity matches
        if top_similarities[0] < 0.1:
            return "Bot: I couldn't find relevant information about that topic in the documents. Try asking about biological neurons, neural networks, or related topics."
        
        # Combine top responses
        responses = []
        for i, (idx, sim) in enumerate(zip(top_indices, top_similarities)):
            if sim > 0.1:  # Only include reasonably similar responses
                responses.append(f"{sentences[idx]}")
        
        if responses:
            combined_response = " ".join(responses[:2])  # Use top 2 responses
            confidence = float(top_similarities[0])
            return f"Bot: {combined_response}\n\n(Confidence: {confidence:.2f})"
        else:
            return "Bot: I couldn't find relevant information about that topic in the documents."
            
    except Exception as e:
        return f"Bot: Sorry, I encountered an error: {str(e)}"

# ---------- Step 7: Streamlit Chat UI ----------
st.title("🧠 Intelligent Deep Learning Chatbot")
st.write("Ask me anything from the uploaded PDFs!")

user_input = st.text_input("You:", "")
if st.button("Ask"):
    if user_input:
        st.write(chatbot_response(user_input))
    else:
        st.write("Please enter a question.")
