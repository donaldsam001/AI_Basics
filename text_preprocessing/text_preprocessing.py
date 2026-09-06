import re
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_cv_text(text: str)-> str:
    if not isinstance(text, str): 
        return ""

    # lower
    text = text.lower()

    # remove URLs, email,...
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)

    # Tokenize by whitespace to preserve technical keywords like C#, C++, Node.js
    tokens = text.split()

    cleaned_tokens = []
    for word in tokens:
        # Strip generic non-alphanumeric at boundaries, but keep # and + for skills
        word = re.sub(r'^[^a-z0-9#+.]+|[^a-z0-9#+.]+$', '', word)
        
        if not word:
            continue
            
        lemma = lemmatizer.lemmatize(word)
        # Removed the len(word) > 1 filter to preserve 'c', 'r', 'f#'
        if lemma not in stop_words:
            cleaned_tokens.append(lemma)

    return " ".join(cleaned_tokens)
