#preprocessing text
import nltk
nltk.download('stopwords')
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re



def preprocess_text(text, max_length, tokenizer=None):
    """
    Preprocesses the text for an LSTM model.

    Parameters:
        text (str): The input text to be preprocessed.
        max_length (int): Maximum length for the padded sequences.
        tokenizer (Tokenizer, optional): If provided, uses this tokenizer; otherwise, fits a new one.

    Returns:
        numpy.array: The preprocessed text sequences.
        Tokenizer: The tokenizer used/fitted on the text.
    """
    # Removing HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Removing punctuation and converting to lowercase
    text = re.sub('[^a-zA-Z\s]', '', text).lower()

    # Removing stopwords
    stops = set(stopwords.words("english"))
    text = " ".join([word for word in text.split() if word not in stops])

    # Tokenizing and converting to sequences
    if tokenizer is None:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts([text])

    sequences = tokenizer.texts_to_sequences([text])

    # Padding the sequences
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

    return padded_sequences, tokenizer

def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub('[^a-zA-Z\s]', '', text).lower()
    stops = set(stopwords.words("english"))
    text = " ".join([word for word in text.split() if word not in stops])
    return text
