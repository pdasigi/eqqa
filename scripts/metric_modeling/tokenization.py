"""Utilities for tokenization."""

from nltk import download
from nltk.corpus import stopwords

download("stopwords")
stop_words = stopwords.words("english")


def remove_punc(text: str) -> str:
    """Minimal processing of punctuation.
    
    It mimics the punctuation handling considered in the MOCHA paper.
    """
    return text.replace('?', '').replace('.', '').replace('!', '')


def normalize_answer(
        s: str,
        no_punct: bool=True,
        lowercase: bool=True,
        white_space_fix: bool=True,
        remove_articles: bool=False,
        remove_stopwords: bool=False,
        is_alpha_only: bool =False,
    ) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    if lowercase:
        s = s.lower()
        
    if no_punct:
        s = remove_punc(s)
    
    if remove_articles:
        import re
        s = re.sub(r"\b(a|an|the)\b", " ", s)

    if remove_stopwords:
        s = [w for w in s.split() if w not in stop_words]
        s = " ".join(s)
        
    if is_alpha_only:
        s = [w for w in s.split() if w.isalpha()]
        s = " ".join(s)
        
    if white_space_fix:
        s = " ".join(s.split())
        
    return s