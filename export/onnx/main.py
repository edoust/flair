from export.inputcreator import InputCreator
from flair.data import Sentence

from torch import torch

def main():
    # Load trained models
    modelName = "flair/upos-multi-fast" # "flair/upos-multi" "flair/upos-multi-fast" # "de-pos"
    sentences = [Sentence("Pla Gon"), Sentence("Xu")]

    # sentences = [Sentence("Ich bin ein selbst für Deutschland außergewöhnlich nüchterner Mensch und verstehe es , meine fünf Sinne zusammenzuhalten .")]

    path = "c:/temp/"

    # Derive from models?
    reverseForwardAndBackward = modelName != "de-pos"
    with_whitespace = modelName != "de-pos" # index of embedding to take varies depending on model

    creator = InputCreator()
    exportModel, forward, forwardIndices, backward, backwardIndices, striping, characterLengths, lengths, forwardDictionary, backwardDictionary, outputTags = creator.createExportModel(modelName, sentences, with_whitespace, reverseForwardAndBackward)

    creator.saveMappingDict(forwardDictionary.item2idx, path, modelName, "fw_item2idx")
    creator.saveMappingDict(backwardDictionary.item2idx, path, modelName, "bw_item2idx")
    creator.saveTags(outputTags, path, modelName)

    exportModel.eval()

    traced_model = torch.jit.trace(exportModel, (
    forward, forwardIndices, backward, backwardIndices, striping, characterLengths, lengths))
    out = traced_model(forward, forwardIndices, backward, backwardIndices, striping, characterLengths, lengths)

    print(out)

    # Export
    in_names = ["forward", "forwardIndices", "backward", "backwardIndices", "striping", "characterLengths", "lengths"]
    out_names = ["scores", "prediction"]

    torch.onnx.export(model=traced_model,
                      args=(forward, forwardIndices, backward, backwardIndices, striping, characterLengths, lengths),
                      f=f'C:/temp/' + modelName + '.onnx',
                      opset_version=16,
                      input_names=in_names,
                      output_names=out_names,
                      verbose=False,
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
