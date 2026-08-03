from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def spam_similarity_score(texts, threshold=0.85):
    """
    Returns:
    - similarity_score (float)
    - clusters (list of lists of indices)
    """

    if len(texts) < 5:
        return 0.0, []

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)

    sim_matrix = cosine_similarity(X)
    np.fill_diagonal(sim_matrix, 0)

    visited = set()
    clusters = []

    for i in range(len(texts)):
        if i in visited:
            continue

        similar = np.where(sim_matrix[i] > threshold)[0].tolist()
        if similar:
            cluster = [i] + similar
            cluster = list(set(cluster))
            visited.update(cluster)

            if len(cluster) > 1:
                clusters.append(cluster)

    similarity_score = sum(len(c) for c in clusters) / len(texts)
    return round(similarity_score, 2), clusters
