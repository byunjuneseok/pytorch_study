import konlpy
from konlpy.tag import Twitter as twit


def tokenizer(tagger, doc):
    """tokenizer with tagger."""
    return ["/".join(p) for p in tagger(doc)]

if __name__ == "__main__":

    """Tagger"""
    tagger = twit()
    print(tagger.pos("역시 파이토치는 재미있네요 ㅋㅋㅋ"))
    print(tagger.nouns("사과는 맛있지만, 바나나는 맛없다"))
    print(tagger.pos("아니, 이렇게 신기할수가?", stem=True))
    print(tokenizer(tagger.pos, "사랑하는 자여 네 영혼이 잘됨 같이 네가 범사에 잘되고 강건하기를"))

