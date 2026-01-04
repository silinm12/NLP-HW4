import json
import re
import gensim
from transformers import pipeline
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from gensim.topic_coherence.indirect_confirmation_measure import word2vec_similarity


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
                clean_text = re.sub(r'[^\w\s]|[\d]','',text)
                #tokenize
                tokens= clean_text.split()
                if tokens:
                    tokenized_sentences.append(tokens)

            except Exception as e:
                continue
    print(f"Done! Created {len(tokenized_sentences)} tokenized sentences.")
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
def find_similar_sentences(sentence,model):
    valid_corpus = []
    for sent in sentence:
        if len([w for w in sent if w in model.wv])>=4:
            valid_corpus.append(sent)
    print(f"Calculating vectors for {len(valid_corpus)} sentences...")
    #choose the first 10 valid sentences
    corpus_vectors = [get_sentence_embedding(s, model) for s in valid_corpus]
    corpus_matrix = np.array(corpus_vectors)

    # We want matches for the first 10 sentences
    target_indices = range(10)

    with open('knesset_similar_sentences.txt', 'w', encoding='utf-8') as f:
        for i in target_indices:
            # Reshape target to shape (1, vector_size)
            target_vec = corpus_matrix[i].reshape(1, -1)

            # Calculate similarity against the ENTIRE corpus at once
            similarities = cosine_similarity(target_vec, corpus_matrix)

            # Set self-similarity to -1 so it doesn't match itself
            similarities[0][i] = -1

            # Find the index of the highest score
            best_idx = np.argmax(similarities)

            target_sent = valid_corpus[i]
            best_match = valid_corpus[best_idx]

            target_str = " ".join(target_sent)
            match_str = " ".join(best_match)
            f.write(f"{target_str}: most similar sentence: {match_str}\n")
            print(f"Finished sentence {i + 1}/10")


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
        replacements_str=",".join(replacements_log)

        line1= f"{i+1}: {original_sent}: {new_sent}"
        line2= f"replaced words: {replacements_str}"
        output_lines.append(line1)
        output_lines.append(line2)

    with open('red_words_sentences.txt','w',encoding='utf-8') as f:
        f.write("\n".join(output_lines))



if __name__ == "__main__":
    corpus_path="knesset_corpus.jsonl"
    #step 1,a: load data
    sentences = load_and_preprocess_corpus(corpus_path)
    #b: train model
    model =Word2Vec(sentences,vector_size=100,window=5,min_count=3)
    model.save('knesset_word2vec.model')
    #c
    word_vectors = model.wv
    print(word_vectors['ישראל'])

    #step 2
    check_similarity(model)
    find_similar_sentences(sentences,model)
    solve_red_words(model)
