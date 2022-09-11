from flair.data import Sentence
from flair.embeddings import FlairEmbeddings
from flair.models import SequenceTagger

from torch import torch

forward = FlairEmbeddings('de-forward')
backward = FlairEmbeddings('de-backward')
sequenceTagger = SequenceTagger.load('de-pos')
lmforward = forward.lm
lmbackward = backward.lm

tagger: SequenceTagger = SequenceTagger.load("de-pos")

# sentence0: Sentence = Sentence("Ich bin ein selbst für Deutschland außergewöhnlich nüchterner Mensch und verstehe es , meine fünf Sinne zusammenzuhalten .")
# sentence1: Sentence = Sentence("Erzählen wir ruhig und ohne alle Aufregung .")

sentence0: Sentence = Sentence("Xu")
sentence1: Sentence = Sentence("Pla Gon")
tagger.predict([sentence1, sentence0])

print("Analysing %s" % sentence0)
print("\nThe following NER tags are found: \n")
print(sentence0.to_tagged_string())

print("Analysing %s" % sentence1)
print("\nThe following NER tags are found: \n")
print(sentence1.to_tagged_string())
