import os
import sys
import shutil

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