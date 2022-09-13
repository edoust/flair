import torch

from torch import nn

import torch.nn.functional as F


class ExportModel(nn.Module):

    def __init__(self,
                 forwardEncoder: nn.Embedding,
                 forwardProj: nn.Linear,
                 forwardNetwork: torch.nn.LSTM,
                 backwardEncoder: nn.Embedding,
                 backwardProj: nn.Linear,
                 backwardNetwork: torch.nn.LSTM,
                 embedding2nn: torch.nn.Linear,
                 rnn: torch.nn.LSTM,
                 linear: torch.nn.Linear):
        super(ExportModel, self).__init__()
        self.forwardEncoder = forwardEncoder
        self.forwardProj = forwardProj
        self.forwardNetwork = forwardNetwork
        self.backwardEncoder = backwardEncoder
        self.backwardProj = backwardProj
        self.backwardNetwork = backwardNetwork
        self.embedding2nn: torch.nn.Linear = embedding2nn
        self.rnn: torch.nn.LSTM = rnn
        self.linear: torch.nn.Linear = linear

    def forward(self, inputForward: torch.Tensor, inputIndicesForward: torch.Tensor, inputBackward: torch.Tensor,
                inputIndicesBackward: torch.Tensor, striping: torch.Tensor, characterLengths: torch.Tensor, lengths: torch.Tensor):

        zeros = torch.zeros(1, lengths.size(0), self.forwardNetwork.hidden_size, dtype=torch.float32)

        # Forward language model
        forwardEncoded = self.forwardEncoder(inputForward)
        packedForward = torch.nn.utils.rnn.pack_padded_sequence(
            forwardEncoded, characterLengths, enforce_sorted=True, batch_first=False
        )
        forwardOutput, forwardHidden = self.forwardNetwork(packedForward, (zeros, zeros))
        forward_result, forward_output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            forwardOutput, batch_first=False
        )

        if self.forwardProj is not None:
            forward_result = self.forwardProj(forward_result)

        # Backward language model
        backwardEncoded = self.backwardEncoder(inputBackward)
        packedBackward = torch.nn.utils.rnn.pack_padded_sequence(
            backwardEncoded, characterLengths, enforce_sorted=True, batch_first=False
        )
        backwardOutput, backwardHidden = self.backwardNetwork(packedBackward, (zeros, zeros))
        backward_result, backward_output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            backwardOutput, batch_first=False
        )

        if self.backwardProj is not None:
            backward_result = self.backwardProj(backward_result)

        # Stack Embeddings
        forward_output_flat = forward_result.view(
            torch.Size([forward_result.size(0) * forward_result.size(1), forward_result.size(2)]))
        forwardEmbeddings = torch.index_select(forward_output_flat, 0, inputIndicesForward)

        backward_output_flat = backward_result.view(
            torch.Size([backward_result.size(0) * backward_result.size(1), backward_result.size(2)]))
        backwardEmbeddings = torch.index_select(backward_output_flat, 0, inputIndicesBackward)

        if self.reverseForwardAndBackward:
            stackedEmbeddings = torch.cat((backwardEmbeddings, forwardEmbeddings), 0)
        else:
            stackedEmbeddings = torch.cat((forwardEmbeddings, backwardEmbeddings), 0)

        stripedEmbeddings = torch.index_select(stackedEmbeddings, 0, striping)

        sentence_tensor = stripedEmbeddings.view(
            torch.Size([inputForward.size(1), torch.max(lengths), 2 * self.embeddingSize]))

        sentence_tensor = self.embedding2nn(sentence_tensor)

        # Run tagging model
        packed = torch.nn.utils.rnn.pack_padded_sequence(sentence_tensor, lengths, enforce_sorted=True,
                                                         batch_first=True)

        # set to 4 in case of upos-multi, 2 otherwise
        hidden = torch.zeros(2, lengths.size(0), self.rnn.hidden_size, dtype=torch.float32)
        rnn_output, hidden = self.rnn(packed, (hidden, hidden))
        sentence_tensor, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)

        features = self.linear(sentence_tensor)

        softmax_batch = F.softmax(features, dim=2)

        scores_batch, prediction_batch = torch.max(softmax_batch, dim=2)

        return scores_batch, prediction_batch
