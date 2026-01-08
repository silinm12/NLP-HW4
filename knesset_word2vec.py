import json
import re
import gensim
import numpy as np
import os
import sys
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

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

#Step 1:
#a
def load_and_preprocess_corpus(file_path):
    tokenized_sentences = []
    with open(file_path, 'r',encoding='utf-8') as f:
        for line in f:
            try:
                data=json.loads(line)
                text = data.get('sentence_text','')
                #clean text from anything that isn't a word, and exchange with ''
                raw_tokens = text.split()
                tokens = [t for t in raw_tokens if any(c.isalnum() for c in t)]
                if tokens:
                    tokenized_sentences.append(tokens)

            except Exception as e:
                continue
    return tokenized_sentences

#step 2:
#a
def check_similarity(model):
    word2_similarity_list=['יום','אישה','דרך','ארוך','תוכנית','אוהב','אסור','איתן','זכות']

    with open('knesset_similar_words.txt','w',encoding='utf-8') as f:
        for word in word2_similarity_list:
            if word in model.wv:
                try:
                    #nearest 5 words
                    neighbors = model.wv.most_similar(word, topn=5)
                    formatted_neighbors = ", ".join([f"('{n[0]}', {n[1]:.4f})" for n in neighbors])
                    f.write(f"{word}: {formatted_neighbors}\n")
                except KeyError:
                    print(f"Error calculating for {word}")
            else:
                print(f"{word} is not in the word2vec model.")

#b: sentence embedding
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

#c:
def find_similar_sentences(corpus_path, model, output_dir):
    #Target the two main speakers for a balanced analysis
    target_speakers = ["ראובן ריבלין", "א' בורג"]
    targets = {sp: [] for sp in target_speakers}
    valid_corpus = []

    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # This call will now work because normalize_speaker is defined
            speaker = normalize_speaker(data.get('speaker_name', ''))
            text = data.get('sentence_text', '')
            tokens = [t for t in text.split() if any(c.isalnum() for c in t)]

            # Only use sentences with at least 4 words known to the model
            if len([w for w in tokens if w in model.wv]) >= 4:
                valid_corpus.append(tokens)
                if speaker in targets and len(targets[speaker]) < 5:
                    targets[speaker].append(tokens)

    target_list = targets[target_speakers[0]] + targets[target_speakers[1]]
    target_vectors = np.array([get_sentence_embedding(s, model) for s in target_list])
    corpus_vectors = np.array([get_sentence_embedding(s, model) for s in valid_corpus])

    output_path = os.path.join(output_dir, 'knesset_similar_sentences.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, target_vec in enumerate(target_vectors):
            similarities = cosine_similarity(target_vec.reshape(1, -1), corpus_vectors)

            target_str = " ".join(target_list[i])
            for idx, corpus_sent in enumerate(valid_corpus):
                if " ".join(corpus_sent) == target_str:
                    similarities[0][idx] = -1
                    break

            best_idx = np.argmax(similarities)
            match_str = " ".join(valid_corpus[best_idx])

            # Formatted exactly per image instructions: <orig>: most similar sentence: <match>
            f.write(f"{target_str}: most similar sentence: {match_str}\n")
#d
def solve_red_words(model):
    sentences_data = [
        ("בשעות הקרובות נפתח את הדיון בסעיף הבא שעל סדר היום", ["בסעיף", "בשעות"]),
        ("אבקש מהנוכחים להתיישב כדי שנוכל להתחיל", ["להתיישב", "אבקש"]),
        ("אנו מודים לצוות המקצועי על עבודתו", ["עבודתו", "לצוות"]),
        ("הנושא יועבר להמשך טיפול בוועדת המשנה", ["טיפול", "הנושא"]),
        ("ההצעה הובאה להצבעה ואושרה", ["ואושרה", "הובאה","ההצעה"])
    ]

    backup_map = {
        'בשעות': 'שעות', 'בסעיף': 'סעיף',
        'אבקש': 'מבקש', 'להתיישב': 'לשבת',
        'לצוות': 'צוות', 'עבודתו': 'עבודה',
        'הנושא': 'נושא', 'טיפול': 'טיפול',
        'הובאה': 'הוגשה', 'ואושרה': 'אושרה',
        "ההצעה":"ההצעה"
    }

    output_lines=[]

    #go over each sentence
    for i, (original_sent,red_words) in enumerate(sentences_data):
        new_sent= original_sent
        replacements_log=[]
        #go over the red words in the current sentence
        for red_word in red_words:
            search_term= red_word
            if red_word not in model.wv:
                #if the red word isnt in the corpus we use the backup
                search_term= backup_map.get(red_word, red_word)

            #if also the backup is not in the model we will skip it
            if search_term not in model.wv:
                print(f"could not find replacement for '{red_word}'")
                replacement= red_word
            else:
                best_match= model.wv.most_similar(search_term, topn=1)[0][0]
                replacement= best_match

            new_sent= new_sent.replace(red_word, replacement)

            replacements_log.append(f"({red_word}:{replacement})")
        replacements_str=" , ".join(replacements_log)

        line1= f"{i+1}: {original_sent}: {new_sent}"
        line2= f"replaced words: {replacements_str}"
        output_lines.append(line1)
        output_lines.append(line2)

    with open('red_words_sentences.txt','w',encoding='utf-8') as f:
        f.write("\n".join(output_lines))



if __name__ == "__main__":
    # Command line argument handling
    if len(sys.argv) != 3:
        print("Usage: python knesset_word2vec.py <path/to/corpus_file.jsonl> <path/to/output_dir>")
        sys.exit(1)

    corpus_path = sys.argv[1]#knesset_corpus.jsonl
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)
    #step 1,a: load data
    sentences = load_and_preprocess_corpus(corpus_path)
    #b: train model
    model =Word2Vec(sentences,vector_size=100,window=5,min_count=3)
    model.save('knesset_word2vec.model')

    #step 2
    check_similarity(model)
    find_similar_sentences(corpus_path, model, output_dir)
    solve_red_words(model)
