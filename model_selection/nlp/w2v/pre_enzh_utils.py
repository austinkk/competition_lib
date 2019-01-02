# -*- coding: utf-8 -*-
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def isSymbol(inputString):
    return bool(re.match(r'[^\w]', inputString))

def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

def getStopWords():
    return stopwords.words('english')

def check(word):
    word = word.lower()
    stop = getStopWords()
    if word in stop:
        return False
    elif hasNumbers(word) or isSymbol(word):
        return False
    else:
        return True

def preprocessing(sen, lang = 0):
    res = []
    if lang == 0:
        wordnet_lemmatizer = WordNetLemmatizer()
        for word in sen:
            if check(word):
                res.append(wordnet_lemmatizer.lemmatize(word))
    else:
        pass
    return res

if __name__ == '__main__':
    print (punctuation)
#string1 = re.sub(r"[%s]+" %punctuation, "",corpus)
#string = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]", "",corpus)
