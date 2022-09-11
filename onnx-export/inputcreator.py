from typing import List

from flair.data import Sentence, Dictionary

import torch

import flair

from flair.models import LanguageModel

class InputCreator():

    def run(self, sentences: List[Sentence], lmforward: LanguageModel, forwardIndices: torch.Tensor, lmbackward: LanguageModel, backwardIndices: torch.Tensor)-> (torch.Tensor, torch.Tensor):

        self.name = 'aef'
        self.chars_per_chunk: int = 512
        self.start_marker = "\n"
        self.end_marker = " "
        self.lm_forward = lmforward
        self.lm_backward = lmbackward
        self.fine_tune = False

        text_sentences = [sentence.to_tokenized_string() for sentence in sentences]
        inputForward = self.getForwardLmInput(text_sentences, self.start_marker, self.end_marker, lmforward.dictionary)
        inputBackward = self.getBackwardLmInput(text_sentences, self.start_marker, self.end_marker, lmbackward.dictionary)
        return (inputForward, inputBackward)

        # This code is ~3 years old and probably doesn't work anymore
        outputBackward = self.runLanguageModel(lmbackward, inputBackward, hidden)
        result = self.postProcessBackwardLanguageModel("back", sentences, outputBackward, True);
        finalOutputBackward = self.postProcessLanguageModel(outputBackward, backwardIndices)
        outputForward = self.runLanguageModel(lmforward, inputForward, hidden)

        finalOutputForward = self.postProcessLanguageModel(outputForward, forwardIndices)

        #self._handle_output_from_lm("name", sentences, inp, True)

        return;


    def getForwardLmInput(self, sentences: List[str], start_marker: str, end_marker: str, dict: Dictionary) -> torch.Tensor:

        longest_padded_str: int = len(max(sentences, key=len)) + len(start_marker) + len(end_marker)

        # pad strings with whitespaces to longest sentence
        padded_sentences: List[str] = []

        for string in sentences:
            padded = f"{start_marker}{string}{end_marker}"
            padded_sentences.append(padded)

        padding_char_index = dict.get_idx_for_item(" ")

        # push batch through the RNN language model
        sequences_as_char_indices: List[List[int]] = []
        for string in padded_sentences:
            char_indices = dict.get_idx_for_items(list(string))
            char_indices += [padding_char_index] * (longest_padded_str - len(string))
            sequences_as_char_indices.append(char_indices)
        batch = torch.tensor(sequences_as_char_indices, dtype=torch.long).to(
            device=flair.device, non_blocking=True
        )

        return batch.transpose(0, 1)

    def getBackwardLmInput(self, sentences: List[str], start_marker: str, end_marker: str, dict: Dictionary) -> torch.Tensor:

        longest_padded_str: int = len(max(sentences, key=len)) + len(start_marker) + len(end_marker)

        # pad strings with whitespaces to longest sentence
        padded_sentences: List[str] = []

        for string in sentences:
            string = string[::-1]
            padded = f"{start_marker}{string}{end_marker}"
            padded_sentences.append(padded)

        padding_char_index = dict.get_idx_for_item(" ")

        # push batch through the RNN language model
        sequences_as_char_indices: List[List[int]] = []
        for string in padded_sentences:
            char_indices = dict.get_idx_for_items(list(string))
            char_indices += [padding_char_index] * (longest_padded_str - len(string))
            sequences_as_char_indices.append(char_indices)
        batch = torch.tensor(sequences_as_char_indices, dtype=torch.long).to(
            device=flair.device, non_blocking=True
        )

        return batch.transpose(0, 1)

    def runLanguageModel(
        self,
        lm,
        batch: torch.Tensor):

        _, rnn_output, hidden = lm.forward(batch)

        return rnn_output

    # this is the flair embeddings part
    def postProcessLanguageModel(self, all_hidden_states_in_lm: torch.Tensor, flatIndices: torch.Tensor) -> List[Sentence]:

        # gradients are enable if fine-tuning is enabled
        gradient_context = torch.enable_grad() if self.fine_tune else torch.no_grad()

        with gradient_context:

            if not self.fine_tune:
                all_hidden_states_in_lm = all_hidden_states_in_lm.detach()

            flat = all_hidden_states_in_lm.view(torch.Size([all_hidden_states_in_lm.size(0) * all_hidden_states_in_lm.size(1), all_hidden_states_in_lm.size(2)]))

            return torch.index_select(flat, 0, flatIndices)


    # this is the flair embeddings part
    def postProcessBackwardLanguageModel(self, name, sentences: List[Sentence], all_hidden_states_in_lm: torch.Tensor, is_forward_lm) -> List[Sentence]:

        # gradients are enable if fine-tuning is enabled
        gradient_context = torch.enable_grad() if self.fine_tune else torch.no_grad()

        with gradient_context:

            if not self.fine_tune:
                all_hidden_states_in_lm = all_hidden_states_in_lm.detach()

            # take first or last hidden states from language model as word representation
            for i, sentence in enumerate(sentences):
                sentence_text = sentence.to_tokenized_string()

                offset_forward: int = len(self.start_marker)
                offset_backward: int = len(sentence_text) + len(self.start_marker)

                for token in sentence.tokens:

                    offset_forward += len(token.text)

                    if is_forward_lm:
                        offset = offset_forward
                    else:
                        offset = offset_backward

                    embedding = all_hidden_states_in_lm[offset, i, :]

                    # if self.tokenized_lm or token.whitespace_after:
                    offset_forward += 1
                    offset_backward -= 1

                    offset_backward -= len(token.text)

                    # only clone if optimization mode is 'gpu'
                    if flair.embedding_storage_mode == "gpu":
                        embedding = embedding.clone()

                    token.set_embedding(self.name, embedding)

            del all_hidden_states_in_lm

        return sentences




