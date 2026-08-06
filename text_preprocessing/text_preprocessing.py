import re
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punlt')
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
    text = re.sub(r'https?://\S+|www\.\S+', text)
    text = re.sub(r'\S+@\S+', '', text)

    # Remove special character, " "
    text = re.sub(r'[^a-z0-9\s+#.]', ' ', text)

    # tokenization
    tokens = word_tokenize(text)

    # Remove Stopwords & Lemmatization
    cleaned_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 1
    ]

    # 
    return " ".join(cleaned_tokens)

