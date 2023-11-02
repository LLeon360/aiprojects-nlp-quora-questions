import torch
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

class EmbeddingDataset(torch.utils.data.Dataset):
    """
    Embeddings Dataset
    """

    def __init__(self, df, vocab, max_seq_length, pad_token, unk_token):
        '''
        data_path (str): path for the csv file that contains the data that you want to use
        '''

        # get labels from label column as list
        self.labels = df.label.tolist()

        # make dictionaries translating from word to index and index to word
        self.word2idx = {term:idx for idx,term in enumerate(vocab)}
        self.idx2word = {idx:word for word,idx in self.word2idx.items()}

        # set pad and unk token for padding and unknown vocab not found in corpus
        self.pad_token,self.unk_token = pad_token,unk_token

        # make input ids are the indices that each word translates, sequence_len is length of sequence excluding padding, label is the label
        self.input_ids = []
        self.sequence_lens = []
        self.labels = []

        self.input_ids, self.sequence_lens = [self.convert_text_to_input_ids(words, max_seq_length) for words in df['question_ids']]
        self.labels = df['target'].tolist()

        #sanity checks
        assert len(self.input_ids) == df.shape[0]
        assert len(self.sequence_lens) == df.shape[0]
        assert len(self.labels) == df.shape[0]

    # return an instance from the dataset
    def __getitem__(self, i):
        '''
        i (int): the desired instance of the dataset
        '''

        # return the ith sample's list of word counts and label
        return self.sequences[i, :], self.labels[i] 

    def convert_text_to_input_ids(self,text,pad_to_len):
        # truncate excess words (beyond the length we should pad to)
        words = text.strip().split()[:pad_to_len]

        # add padding till we've reached desired length
        deficit = pad_to_len - len(words)
        words.extend([self.pad_token]*deficit)

        # replace words with their id
        for i in range(len(words)):
            if words[i] not in self.word2idx:
                # if word is not in vocab, then use <unk> token
                words[i] = self.word2idx[self.unk_token]
            else:
                # else find the id associated with the word
                words[i] = self.word2idx[words[i]]
        return torch.Tensor(words).long(),pad_to_len - deficit

    # return the size of the dataset
    def __len__(self):
        # Make dataset compatible with len() function
        return len(self.input_ids)