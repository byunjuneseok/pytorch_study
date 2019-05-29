import codecs
import pandas as pd

from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer

import torch
import torch.nn as nn

import numpy as np
import random

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


if __name__ == "__main__":

    f = codecs.open("data/reviews.txt", 'r', 'utf-8')
    # f.read()[:500]

    keywords = ["발열", "소음"]
    keywords_english = ["heat", "noise"]
    for i, keyword in enumerate(keywords):
        temp_list = []
        save_name = "data/reviews_" + keywords_english[i] + ".txt"
        f = codecs.open("data/reviews.txt", 'r', 'utf-8')
        t = codecs.open(save_name, 'w', 'utf-8')

        while True:
            line = f.readline()
            if not line: break
            if keyword in line:
                temp_list.append(line)

        set_list = list(set(temp_list))

        for item in set_list:
            t.write(item)

        f.close()
        t.close()

    f = codecs.open("data/reviews_heat.txt", 'r', 'utf-8')

    # f.read()[:500]
