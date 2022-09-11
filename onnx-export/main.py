from flair.data import Sentence
from flair.models import SequenceTagger, LanguageModel
from flair.embeddings import FlairEmbeddings

from torch import torch

from inputcreator import InputCreator

from exportmodel import ExportModel

def main():

    #tagger = SequenceTagger.load('de-pos')

    ## Load trained models
    forward = FlairEmbeddings('de-forward')
    backward = FlairEmbeddings('de-backward')
    sequenceTagger = SequenceTagger.load('de-pos')
    lmforward = forward.lm
    lmbackward = backward.lm


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
    zeros = torch.zeros([1, 2, 2048], dtype=torch.float32)
    hidden = torch.zeros([2, 2, 256], dtype=torch.float32)
    inputForward, inputBackward = creator.run([sentence0, sentence1], lmforward, forwardIndices, lmbackward, forwardIndices, zeros)
    tokenShape = torch.zeros([2], dtype=torch.int64)

    ## Execute
    exportModel = ExportModel(lmforward.encoder,
                                    lmforward.rnn,
                                    lmbackward.encoder,
                                    lmbackward.rnn,
                                    sequenceTagger.embedding2nn, sequenceTagger.rnn, sequenceTagger.linear)
    prediction = exportModel.forward(inputForward, forwardIndices, inputBackward, backwardIndices, striping, characterLengths, lengths, zeros, hidden, tokenShape)

    print(prediction)

    #return
    ## Export
    in_names = ["inputForward", "inputForwardIndices", "inputBackward", "inputBackwardIndices", "striping", "characterLengths", "lengths", "zeros", "hidden", "tokenShape"]
    out_names = ["scores", "prediction"]

    torch.onnx.export(model=exportModel,
                args=(inputForward, forwardIndices, inputBackward, backwardIndices, striping, characterLengths, lengths, zeros, hidden, tokenShape),
                f=f'C:/temp/export.onnx',
                opset_version=8,
                input_names=in_names,
                output_names=out_names,
                verbose=True,
                dynamic_axes=
                {
                    'inputForward': { 0:'characters', 1:'sentences' },
                    'inputForwardIndices': { 0:'total_tokens' },
                    'inputBackward': { 0:'characters', 1:'sentences' },
                    'inputBackwardIndices': { 0:'total_tokens' },
                    'striping': { 0:'total_embeddings' },
                    'characterLengths': { 0:'sentences' },
                    'lengths': { 0:'sentences' },
                    'zeros': { 1:'sentences' },
                    'hidden': { 1:'sentences' },
                    'tokenShape': { 0:'tokens' },
                    'scores': { 0:'sentences', 1:'tokens' },
                    'prediction': { 0:'sentences', 1:'tokens' }
                    #'scores': { 0:'sentences', 1: 'tokens' }
                })






if __name__ == '__main__':

    main()