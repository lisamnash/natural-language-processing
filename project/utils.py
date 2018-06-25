import nltk
import pickle
import re
import numpy as np
import csv

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': './data/starspace_embedding.tsv',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """
    
    # Hint: you have already implemented a similar routine in the 3rd assignment.
    # Note that here you also need to know the dimension of the loaded embeddings.
    # When you load the embeddings, use numpy.float32 type as dtype

    import numpy as np
    starspace_embedding = {}
    import csv
    with open(embeddings_path, newline='') as embedding_file:
        reader = csv.reader(embedding_file, delimiter='\t')
        embedding_file_lines = list(reader)
        for line in embedding_file_lines:
            word = line[0]
            embedding = np.array(line[1:]).astype(np.float32)
            starspace_embedding[word] = embedding
    dimension = len(line)-1
    return starspace_embedding, dimension




def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings."""

    """
            question: a string
            embeddings: dict where the key is a word and a value is its' embedding
            dim: size of the representation

            result: vector representation for the question
        """
    words = question.split()  # i'll just assume that we have no punctuation or caps we need to deal with)
    embedding_vecs = np.array([embeddings[word] for word in words if word in embeddings])

    if len(embedding_vecs) > 0:
        return np.mean(embedding_vecs, axis=0)
    else:
        return np.zeros(dim)


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
