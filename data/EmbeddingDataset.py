import torch
import pandas as pd
import numpy as np

class EmbeddingDataset(torch.utils.data.Dataset):
    """
    Embeddings Dataset
    """

    def __init__(self, df, vocab, max_seq_length, pad_token, unk_token, test_dataset = False):
        '''
        data_path (str): path for the csv file that contains the data that you want to use
        '''
        # make dictionaries translating from word to index and index to word
        self.word2idx = {term:idx for idx,term in enumerate(vocab)}
        self.idx2word = {idx:word for word,idx in self.word2idx.items()}

        # set pad and unk token for padding and unknown vocab not found in corpus
        self.pad_token,self.unk_token = pad_token,unk_token

        # make input ids are the indices that each word translates, sequence_len is length of sequence excluding padding, label is the label
        self.input_ids = []
        self.sequence_lens = []

        count = 0
        for words in df['question_text']:
            input_ids, seq_len = self.convert_text_to_input_ids(words, max_seq_length)
            self.input_ids.append(input_ids)
            self.sequence_lens.append(seq_len)

        # get labels from label column as list
        if(test_dataset):
            self.labels = [0] * df['question_text'].shape[0]
        else:
            self.labels = df['target'].tolist()

        #sanity checks
        assert len(self.input_ids) == df.shape[0]
        assert len(self.sequence_lens) == df.shape[0]
        assert len(self.labels) == df.shape[0]

    def __getitem__(self, i):
        # for the ith indexm return a dictionary containing id, length and label
        sample_dict = dict()
        sample_dict['input_ids'] = self.input_ids[i].reshape(-1)
        sample_dict['sequence_len'] = torch.tensor(self.sequence_lens[i]).long()
        sample_dict['labels'] = torch.tensor(self.labels[i]).type(torch.FloatTensor)
        return sample_dict

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