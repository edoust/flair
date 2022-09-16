from typing import List

from export.exportmodel import ExportModel
from flair.data import Sentence, Dictionary

import torch

import flair

import re

from flair.models import LanguageModel, SequenceTagger


class InputCreator:
    def createExportModel(
        self, name: str, sentences: List[Sentence], with_whitespace: bool, reverseForwardAndBackward: bool
    ):

        sequenceTagger: SequenceTagger = SequenceTagger.load(name)
        forward = sequenceTagger.embeddings.list_embedding_0
        backward = sequenceTagger.embeddings.list_embedding_1
        lmforward = forward.lm
        lmbackward = backward.lm
        embeddingSize = forward.embedding_length

        # Prepare
        forward, forwardIndices, backward, backwardIndices, lengths, characterLengths = self.run(
            sentences, lmforward, lmbackward, with_whitespace
        )

        # Execute
        exportModel = ExportModel(
            lmforward.encoder,
            lmforward.proj,
            lmforward.rnn,
            lmbackward.encoder,
            lmbackward.proj,
            lmbackward.rnn,
            sequenceTagger.embedding2nn,
            sequenceTagger.rnn,
            sequenceTagger.linear,
        )
        exportModel.embeddingSize = embeddingSize
        exportModel.reverseForwardAndBackward = reverseForwardAndBackward

        return (
            exportModel,
            forward,
            forwardIndices,
            backward,
            backwardIndices,
            characterLengths,
            lengths,
            lmforward.dictionary,
            lmbackward.dictionary,
            sequenceTagger.label_dictionary.idx2item,
        )

    def run(self, sentences: List[Sentence], lmforward: LanguageModel, lmbackward: LanguageModel, withWhitespace: bool):

        self.name = "aef"
        self.chars_per_chunk: int = 512
        self.start_marker = "\n"
        self.end_marker = " "
        self.lm_forward = lmforward
        self.lm_backward = lmbackward
        self.fine_tune = False

        text_sentences = [sentence.to_tokenized_string() for sentence in sentences]
        inputForward = self.getForwardLmInput(text_sentences, self.start_marker, self.end_marker, lmforward.dictionary)
        inputBackward = self.getBackwardLmInput(
            text_sentences, self.start_marker, self.end_marker, lmbackward.dictionary
        )

        lengths = torch.LongTensor([len(x.tokens) for x in sentences])

        characterLengths = torch.LongTensor([2 + len(x.text) for x in sentences])  # forward: +2

        forwardIndices = []
        backwardIndices = []

        sentenceIndex = 0
        sentenceCount = len(sentences)
        maxTokenCount = max(lengths)
        lastCharIndex = 1 + max([len(x.text) for x in sentences])
        for sentence in sentences:

            pos = 0
            if withWhitespace:
                pos += 1

            for token in sentence.tokens:
                pos += len(token.text)
                forwardIndices.append(pos * sentenceCount + sentenceIndex)
                pos += 1

            if maxTokenCount > len(sentence.tokens):
                for i in range(len(sentence.tokens), maxTokenCount):
                    forwardIndices.append(lastCharIndex * sentenceCount + sentenceIndex)

            sentenceIndex += 1

        sentenceIndex = 0
        for sentence in sentences:

            pos = len(sentence.text)
            if withWhitespace:
                pos += 1

            for token in sentence.tokens:
                backwardIndices.append(pos * sentenceCount + sentenceIndex)
                pos -= len(token.text)
                pos -= 1

            if maxTokenCount > len(sentence.tokens):
                for i in range(len(sentence.tokens), maxTokenCount):
                    backwardIndices.append(lastCharIndex * sentenceCount + sentenceIndex)

            sentenceIndex += 1

        return (
            inputForward,
            torch.LongTensor(forwardIndices),
            inputBackward,
            torch.LongTensor(backwardIndices),
            lengths,
            characterLengths,
        )

    def saveMappingDict(self, dict, path: str, modelName: str, dictName: str):
        texts = []
        line = "{char}, {value} }}"
        for x, i in enumerate(dict):
            texts.append(("{ " + (line.format(char=i, value=x))[1:]).replace("\\x", "0x"))

        text = "{\n\t" + ",\n\t".join(texts) + "\n}"

        text = re.sub(r"\'0x(\w\w)0x(\w\w)\'", r"ConvertToUnicodeCharacter(0x\1, 0x\2)", text)
        text = re.sub(r"\'0x(\w\w)0x(\w\w)0x(\w\w)\'", r"ConvertToUnicodeCharacter(0x\1, 0x\2, 0x\3)", text)

        text_file = open(path + modelName + ".onnx." + dictName, "w")
        text_file.write(text)
        text_file.close()

        # private static char ConvertToUnicodeCharacter(int a, int b)
        # {
        #     return Encoding.UTF8.GetChars(new[] { (byte)a, (byte)b }).Single();
        # }

        # private static char ConvertToUnicodeCharacter(int a, int b, int c)
        # {
        #     return Encoding.UTF8.GetChars(new[] { (byte)a, (byte)b, (byte)c }).Single();
        # }

    def saveTags(self, tags, path: str, modelName: str):
        texts = []
        for x, i in enumerate(tags):
            texts.append(i.decode("utf-8"))

        text = '{\n\t"' + '",\n\t"'.join(texts) + '"\n}'

        text_file = open(path + modelName + ".onnx." + "tags", "w")
        text_file.write(text)
        text_file.close()

    def getForwardLmInput(
        self, sentences: List[str], start_marker: str, end_marker: str, dict: Dictionary
    ) -> torch.Tensor:

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
        batch = torch.tensor(sequences_as_char_indices, dtype=torch.long).to(device=flair.device, non_blocking=True)

        return batch.transpose(0, 1)

    def getBackwardLmInput(
        self, sentences: List[str], start_marker: str, end_marker: str, dict: Dictionary
    ) -> torch.Tensor:

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
        batch = torch.tensor(sequences_as_char_indices, dtype=torch.long).to(device=flair.device, non_blocking=True)

        return batch.transpose(0, 1)

    def runLanguageModel(self, lm, batch: torch.Tensor):

        _, rnn_output, hidden = lm.forward(batch)

        return rnn_output

    # this is the flair embeddings part
    def postProcessLanguageModel(
        self, all_hidden_states_in_lm: torch.Tensor, flatIndices: torch.Tensor
    ) -> List[Sentence]:

        # gradients are enable if fine-tuning is enabled
        gradient_context = torch.enable_grad() if self.fine_tune else torch.no_grad()

        with gradient_context:
            if not self.fine_tune:
                all_hidden_states_in_lm = all_hidden_states_in_lm.detach()

            flat = all_hidden_states_in_lm.view(
                torch.Size(
                    [all_hidden_states_in_lm.size(0) * all_hidden_states_in_lm.size(1), all_hidden_states_in_lm.size(2)]
                )
            )

            return torch.index_select(flat, 0, flatIndices)

    # this is the flair embeddings part
    def postProcessBackwardLanguageModel(
        self, name, sentences: List[Sentence], all_hidden_states_in_lm: torch.Tensor, is_forward_lm
    ) -> List[Sentence]:

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
