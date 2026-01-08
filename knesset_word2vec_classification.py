import json
import re
import numpy as np
import random
import sys
import os
from collections import Counter, defaultdict
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from gensim.models import Word2Vec

# --- Helper Functions from HW3 ---

def iter_jsonl(path):
    """Reads a jsonl file line by line to save memory."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def count_speakers(corpus_path):
    """Counts the number of sentences per speaker in the jsonl file."""
    speaker_counter = Counter()
    for obj in iter_jsonl(corpus_path):
        speaker = obj.get("speaker_name")
        if speaker:
            speaker_counter[speaker] += 1
    return speaker_counter

def normalize_speaker(speaker):
    """Maps speaker variants to a canonical name."""
    SPEAKER_ALIASES = {
        "ראובן ריבלין": "ראובן ריבלין",
        "ר' ריבלין": "ראובן ריבלין",
        "רובי ריבלין": "ראובן ריבלין",
        "א' בורג": "א' בורג",
        "אברהם בורג": "א' בורג",
    }
    return SPEAKER_ALIASES.get(speaker, speaker)

def get_top_two_speakers(speaker_counts):
    """Selects top 2 non-system speakers for the binary task."""
    META_SPEAKERS = {"רישום פרלמנטרי", "חברי הכנסת"}
    filtered = Counter({
        speaker: count
        for speaker, count in speaker_counts.items()
        if speaker not in META_SPEAKERS
    })
    return filtered.most_common(2)

def build_binary_dataset(corpus_path, first_speaker, second_speaker):
    """Creates a binary dataset for the two target speakers."""
    x_binary, y_binary = [], []
    for obj in iter_jsonl(corpus_path):
        raw_speaker = obj.get("speaker_name")
        sentence = obj.get("sentence_text")
        if not raw_speaker or not sentence:
            continue
        speaker = normalize_speaker(raw_speaker)
        if speaker == first_speaker:
            x_binary.append(sentence)
            y_binary.append("first")
        elif speaker == second_speaker:
            x_binary.append(sentence)
            y_binary.append("second")
    return x_binary, y_binary

def downsample_binary(X, y, seed=42):
    """Balances binary classes by random down-sampling."""
    random.seed(seed)
    groups = defaultdict(list)
    for sent, label in zip(X, y):
        groups[label].append(sent)
    min_size = min(len(groups["first"]), len(groups["second"]))
    X_balanced, y_balanced = [], []
    for label in ["first", "second"]:
        sampled = random.sample(groups[label], min_size)
        X_balanced.extend(sampled)
        y_balanced.extend([label] * min_size)
    return X_balanced, y_balanced

#Step 2b:
def get_sentence_embedding(tokens, model):
    """Calculates sentence embedding by averaging vectors of known words."""
    valid_words = [word for word in tokens if word in model.wv]
    if not valid_words:
        return np.zeros(model.vector_size)
    vectors = [model.wv[word] for word in valid_words]
    return np.mean(vectors, axis=0)

def vectorize_list_to_w2v(sentences_list, model):
    """Converts a list of sentence strings into a 2D array of embeddings."""
    features = []
    for sent_str in sentences_list:
        # Consistency with Step A: Split by whitespace and keep tokens with alphanumeric chars
        raw_tokens = sent_str.split()
        tokens = [t for t in raw_tokens if any(c.isalnum() for c in t)]
        vec = get_sentence_embedding(tokens, model)
        features.append(vec)
    return np.array(features)

if __name__ == "__main__":
    # Check for correct command line arguments
    if len(sys.argv) != 3:
        print("Usage: python knesset_word2vec_classification.py <corpus_file.jsonl> <knesset_word2vec.model>")
        sys.exit(1)

    corpus_path = sys.argv[1]
    model_path = sys.argv[2]

    if not os.path.exists(corpus_path) or not os.path.exists(model_path):
        print("Error: Input files not found.")
        sys.exit(1)

    # Load the Word2Vec model trained in Step A
    model = Word2Vec.load(model_path)

    # Prepare Binary Classification Dataset
    speaker_counts = count_speakers(corpus_path)
    (first_spk, _), (second_spk, _) = get_top_two_speakers(speaker_counts)
    X_bin, y_bin = build_binary_dataset(corpus_path, first_spk, second_spk)

    # Balance classes and shuffle
    X_bal, y_bal = downsample_binary(X_bin, y_bin)
    X_bal, y_bal = shuffle(X_bal, y_bal, random_state=42)

    # Vectorize: Convert text strings into dense Word2Vec vectors
    X_features = vectorize_list_to_w2v(X_bal, model)

    # Initialize KNN classifier (k=5 as per HW3)
    knn = KNeighborsClassifier(n_neighbors=5)

    # Perform 5-fold Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(knn, X_features, y_bal, cv=cv)

    # Generate and print the classification report
    report = classification_report(y_bal, y_pred, digits=3)
    print("\n" + "=" * 60)
    print(f"Task: Binary Classification ({first_spk} vs {second_spk})")
    print("Features: Word2Vec Sentence Embeddings")
    print(report)
    print("=" * 60)