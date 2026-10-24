"""Text preprocessing: normalise, tokenise, remove stop words, lemmatise."""
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}


def ensure_nltk_data() -> None:
    """Download required NLTK data once if it is missing."""
    for name, path in _RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


ensure_nltk_data()

_lemmatizer = WordNetLemmatizer()
# Keep negations and question words: they carry intent ("cancel" vs "do not cancel").
_KEEP = {"no", "not", "nor", "cannot", "what", "where", "when", "how", "why", "can"}
_STOP_WORDS = set(stopwords.words("english")) - _KEEP


def clean_text(text: str) -> str:
    """Lowercase, expand a few contractions, strip punctuation and extra spaces."""
    text = text.lower().strip()
    text = text.replace("can't", "cannot").replace("won't", "will not")
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_tokens(text: str) -> list[str]:
    """Full pipeline -> list of lemmatised tokens."""
    tokens = word_tokenize(clean_text(text))
    tokens = [t for t in tokens if t not in _STOP_WORDS]
    return [_lemmatizer.lemmatize(t) for t in tokens]


def preprocess(text: str) -> str:
    """Full pipeline -> single string (what the TF-IDF vectoriser consumes)."""
    return " ".join(preprocess_tokens(text))
