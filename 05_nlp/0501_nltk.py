import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tag import pos_tag

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

from collections import Counter
import numpy as np



def tokenizer(tagger, doc):
    """tokenizer with tagger."""
    return ["/".join(p) for p in tagger(doc)]


def DrawWordCloud(data):
    wordcloud = WordCloud(stopwords = STOPWORDS,
                          background_color = 'white', width= 800, height = 400).generate(data)
    plt.figure(figsize = (15 , 12))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


 
if __name__ == "__main__":

    """Download textfiles and read one file"""
    nltk.download("gutenberg")
    nltk.download("stopwords")
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    # print(nltk.corpus.gutenberg.fileids())

    """Read files and convert upper characters to lower."""
    text = nltk.corpus.gutenberg.raw("shakespeare-macbeth.txt")
    text = text.lower()

    """Tokenize word with regular expression"""
    # print(word_tokenize(text[:500]))
    tokens = RegexpTokenizer("[\w]+").tokenize(text[:500])

    """Stopping word"""
    stopping = set(stopwords.words('english'))
    # print(stopping)
    # print('the' in stopping)    # True
    # print([token for token in tokens if not token in stopping])

    """Stemming and tagging"""
    print([PorterStemmer().stem(token) for token in tokens])
    print(pos_tag(tokens))
    print(tokenizer(pos_tag, tokens))

    """Wordcloud"""
    #DrawWordCloud(text)

    """Count word"""
    counts = dict(Counter(RegexpTokenizer("[\w]+").tokenize(text)).most_common(20))
    # Zip keys = labels / values = values
    labels, values = zip(*counts.items())

    indSort = np.argsort(values)[::-1]

    labels = np.array(labels)[indSort]
    values = np.array(values)[indSort]

    indexes = np.arange(len(labels))


    plt.figure(figsize=(15,5))

    plt.xlabel('Top 15 Words')
    plt.ylabel('Frequency')
    plt.title('Word Frequency')

    plt.bar(indexes, values)
    plt.xticks(indexes, labels)
    plt.show()