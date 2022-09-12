from export.inputcreator import InputCreator
from flair.data import Sentence

from torch import torch

def main():
    # Load trained models
    modelName = "flair/upos-multi-fast"  # "de-pos"
    sentences = [Sentence("Pla Gon"), Sentence("Xu")]

    creator = InputCreator()
    exportModel, forward, forwardIndices, backward, backwardIndices, striping, characterLengths, lengths = creator.createExportModel(modelName, sentences)
    prediction = exportModel.forward(forward, forwardIndices, backward, backwardIndices, striping, characterLengths, lengths)

    print(prediction)

    traced_model = torch.jit.trace(exportModel, (
    forward, forwardIndices, backward, backwardIndices, striping, characterLengths, lengths))
    out = traced_model(forward, forwardIndices, backward, backwardIndices, striping, characterLengths, lengths)

    print(out)

if __name__ == '__main__':
    main()
