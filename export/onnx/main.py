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

    # Export
    in_names = ["forward", "forwardIndices", "backward", "backwardIndices", "striping", "characterLengths", "lengths"]
    out_names = ["scores", "prediction"]

    torch.onnx.export(model=exportModel,
                      args=(forward, forwardIndices, backward, backwardIndices, striping, characterLengths, lengths),
                      f=f'C:/temp/' + modelName + '.onnx',
                      opset_version=16,
                      input_names=in_names,
                      output_names=out_names,
                      verbose=True,
                      dynamic_axes=
                      {
                          'forward': {0: 'characters', 1: 'sentences'},
                          'forwardIndices': {0: 'total_tokens'},
                          'backward': {0: 'characters', 1: 'sentences'},
                          'backwardIndices': {0: 'total_tokens'},
                          'striping': {0: 'total_embeddings'},
                          'characterLengths': {0: 'sentences'},
                          'lengths': {0: 'sentences'},
                          'scores': {0: 'sentences', 1: 'tokens'},
                          'prediction': {0: 'sentences', 1: 'tokens'}
                      })


if __name__ == '__main__':
    main()
