import numpy as np
import nltk
#nltk.download('punkt')


from nltk.stem.porter import PorterStemmer
stemmer=PorterStemmer()

def token_ize(sentence):
    # tokenize each sentence by splitting from " " 
    return nltk.word_tokenize(sentence)


def stemming(word):
    return stemmer.stem(word.lower())

def b_o_w(tokenize_sentence,list_of_words):
    """
    Parameters
    ----------
    tokenize_sentence : ["hi","how","are","you"]
        result of tokenizer 
    list_of_words : ["hi","bye","how","are","you"]
        bag of all words
    Returns
    -------
    list with [1,0,1,1,1]
    """
    stemmed=[stemming(w) for w in tokenize_sentence]
    bag=np.zeros(len(list_of_words),dtype=np.float32)
    for idx,word in enumerate(list_of_words):
        if word in stemmed:
            bag[idx]=1.0
    return bag
    


    
    