from flair.data import Sentence
from flair.models import SequenceTagger, LanguageModel
from flair.embeddings import FlairEmbeddings

from torch import torch

from inputcreator import InputCreator

from exportmodel import ExportModel

def main():

    # Load trained models
    modelName = "flair/upos-multi-fast" # "de-pos"

    sequenceTagger: SequenceTagger = SequenceTagger.load(modelName)
    forward = sequenceTagger.embeddings.list_embedding_0;
    backward = sequenceTagger.embeddings.list_embedding_1;

    lmforward = forward.lm
    lmbackward = backward.lm

    embeddingSize = forward.embedding_length

    ## Prepare
    creator = InputCreator()

    sentence0 : Sentence = Sentence("Pla Gon")
    sentence1 : Sentence = Sentence("Xu")

    #sequenceTagger.predict([sentence0, sentence1])
    #return

    lengths = torch.LongTensor([2,2])
    characterLengths = torch.LongTensor([9,4]) # forward: +2
    forwardIndices = torch.LongTensor([3 * 2, 7 * 2, 2 * 2 + 1, 8 * 2 + 1])
    # forwardIndices = torch.LongTensor([3 * 2, 9 * 2, 4 * 2 + 1, 8 * 2 + 1])
    backwardIndices = torch.LongTensor([7 * 2, 3 * 2, 2 * 2 + 1, 8 * 2 + 1])
    # backwardIndices = torch.LongTensor([6 * 2, 3 * 2, 8 * 2 + 1, 4 * 2 + 1])
    striping = torch.LongTensor([0, 4, 1, 5, 2, 6, 3, 7])
    forward, backward = creator.run([sentence0, sentence1], lmforward, forwardIndices, lmbackward, forwardIndices)

    ## Execute
    exportModel = ExportModel(lmforward.encoder,
                                    lmforward.proj,
                                    lmforward.rnn,
                                    lmbackward.encoder,
                                    lmbackward.proj,
                                    lmbackward.rnn,
                                    sequenceTagger.embedding2nn,
                                    sequenceTagger.rnn,
                                    sequenceTagger.linear)
    exportModel.embeddingSize = embeddingSize
    prediction = exportModel.forward(forward, forwardIndices, backward, backwardIndices, striping, characterLengths, lengths)

    print(prediction)

    #return
    ## Export
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
                    'forward': { 0:'characters', 1:'sentences' },
                    'forwardIndices': { 0:'total_tokens' },
                    'backward': { 0:'characters', 1:'sentences' },
                    'backwardIndices': { 0:'total_tokens' },
                    'striping': { 0:'total_embeddings' },
                    'characterLengths': { 0:'sentences' },
                    'lengths': { 0:'sentences' },
                    'scores': { 0:'sentences', 1:'tokens' },
                    'prediction': { 0:'sentences', 1:'tokens' }
                })






if __name__ == '__main__':

    main()