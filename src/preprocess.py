import re


def clean_text(text: str) -> str:
    """Lightweight normalization used for similarity/spam-cluster detection.
    The analyzer works on the *original* text (see analyzer.py) so emotive
    punctuation/case isn't lost for sentiment — this cleaned version is only
    for TF-IDF similarity and burst clustering.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()
