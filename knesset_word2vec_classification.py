import json
import re
import numpy as np
import random
from collections import Counter, defaultdict
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle

#step 3

#funtions from HW3
def count_speakers(corpus_path):
    # Counts the number of sentences per speaker in the jsonl file
    speaker_counter = Counter()
    for obj in iter_jsonl(corpus_path):
        speaker = obj.get("speaker_name")
        if speaker:
            speaker_counter[speaker] += 1
    return speaker_counter

def iter_jsonl(path):
    # Reads the jsonl file line by line to save memory
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def normalize_speaker(speaker):
    # Maps speaker variants to a canonical name (e.g., Rivlin variants)
    SPEAKER_ALIASES = {
        "ראובן ריבלין": "ראובן ריבלין",
        "ר' ריבלין": "ראובן ריבלין",
        "רובי ריבלין": "ראובן ריבלין",
        "א' בורג": "א' בורג",
        "אברהם בורג": "א' בורג",
    }
    return SPEAKER_ALIASES.get(speaker, speaker)

def get_top_two_speakers(speaker_counts):
    # Selects top 2 non-system speakers for the binary task
    META_SPEAKERS = {"רישום פרלמנטרי", "חברי הכנסת"}
    filtered = Counter({
        speaker: count
        for speaker, count in speaker_counts.items()
        if speaker not in META_SPEAKERS
    })
    return filtered.most_common(2)

def build_step1_datasets(corpus_path, first_speaker, second_speaker):
    # Creates binary and multi-class datasets from the corpus
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
    return (x_binary, y_binary), (None, None) # Multi-class omitted for brevity

def downsample_binary(X, y, seed=42):
    # Balances binary classes by random down-sampling
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

#step 2.b
def get_sentence_embedding(sentence,model):
    #ignore words that are OOV
    valid_words = [word for word in sentence if word in model.wv]

    #if there are 0 words that are valid
    if not valid_words:
        return np.zeros(model.vector_size)

    #list of vectors
    vectors = [model.wv[word] for word in valid_words]

    #calculate avg
    sentence_embedding = np.mean(vectors, axis=0)
    return sentence_embedding



def vectorize_list_to_w2v(sentences_list, model):
    # This turns a list of sentence strings into a 2D numpy array of embeddings
    features = []
    for sent_str in sentences_list:
        # Clean and tokenize using the same logic as Step 1
        tokens = re.sub(r'[^\w\s]|[\d]', '', sent_str).split()
        # Use your Step 2b function to get the average vector
        vec = get_sentence_embedding(tokens, model)
        features.append(vec)
    return np.array(features)


if __name__ == "__main__":
    corpus_path = "knesset_corpus.jsonl"
    model= "knesset_word2vec.model"
    #Load data for the binary task (Top 2 speakers)
    speaker_counts = count_speakers(corpus_path)
    (first_spk, _), (second_spk, _) = get_top_two_speakers(speaker_counts)
    (X_bin, y_bin), _ = build_step1_datasets(corpus_path, first_spk, second_spk)

    #Balance classes and shuffle
    X_bal, y_bal = downsample_binary(X_bin, y_bin)
    X_bal, y_bal = shuffle(X_bal, y_bal, random_state=42)

    #Vectorize: Convert text strings into dense Word2Vec vectors
    print(f"Vectorizing {len(X_bal)} sentences using Word2Vec...")
    X_features = vectorize_list_to_w2v(X_bal, model)

    #Initialize the same KNN model from HW3 (k=5)
    knn = KNeighborsClassifier(n_neighbors=5)

    #Perform 5-fold Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("Running 5-fold Cross-Validation...")
    y_pred = cross_val_predict(knn, X_features, y_bal, cv=cv)

    #Generate and save report
    report = classification_report(y_bal, y_pred, digits=3)
    print("=" * 60)
    print(f"Task: Binary Classification ({first_spk} vs {second_spk})")
    print("Features: Word2Vec Sentence Embeddings")
    print(report)

    with open('step3_classification_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print("Step 3 report saved to 'step3_classification_report.txt'")