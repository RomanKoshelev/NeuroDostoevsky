import os
import sys
import shutil
import nltk


def make_dir(path: str):
    folder = os.path.dirname(path)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return path


def remove_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)

        
def prepare_sample(s):
    s = s.replace(' .', '.').replace(' ,', ',').replace(' !', '!').replace(' ?', '?').replace(' :', ':')
    s = s.replace('— —', '—').replace(',,', ',').replace(',.', ',').replace('« ', '«').replace(' »', '»')
    s = s.replace('—,', '—')
    return s


def sent_to_words(sent):
    IGNORE = "[]{}<>~@#$%^/\|_+*…«»\"\'" #()–:
    s = sent.lower()
    for c in IGNORE:
        s=s.replace(c, ' ')
    return [w for w in nltk.tokenize.word_tokenize(s)]


def words_to_sent(words):
    s = ' '.join(words)
    s = s.replace(' ,', ',').replace(' !', '!').replace(' ?', '?').replace(' .', '.').replace(' :', ':')
    s = s.replace(' )', ')').replace('( ', '(').replace(' ;', ';')
    s = s[0].upper() + s[1:]
    return s