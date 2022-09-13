from flair.data import Sentence
from flair.embeddings import FlairEmbeddings
from flair.models import SequenceTagger

from torch import torch

# forward = FlairEmbeddings('de-forward')
# backward = FlairEmbeddings('de-backward')
# sequenceTagger = SequenceTagger.load('de-pos')

tagger: SequenceTagger = SequenceTagger.load("flair/upos-multi-fast")
print(tagger)

sentence0: Sentence = Sentence("Xu")
sentence1: Sentence = Sentence("Pla Gon")

# sentence0: Sentence = Sentence("Erzählen wir ruhig und ohne alle Aufregung .")
sentence1: Sentence = Sentence("Ich bin ein selbst für Deutschland außergewöhnlich nüchterner Mensch und verstehe es , meine fünf Sinne zusammenzuhalten .")
# sentence2: Sentence = Sentence("We define General Aviation as any fixed wing or rotor wing aircraft fitted with 50 passenger seats or less being used for any purpose .")

tagger.predict([sentence1])

# print("Analysing %s" % sentence0)
# print("\nThe following NER tags are found: \n")
# print(sentence0.to_tagged_string())

print("Analysing %s" % sentence1)
print("\nThe following NER tags are found: \n")
print(sentence1.to_tagged_string())
