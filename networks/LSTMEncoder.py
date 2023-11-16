import torch

class LSTMEncoder(torch.nn.Module):
    def __init__(self, config):
        super(LSTMEncoder, self).__init__()

        self.device = config['device']

        # use the pretrained embeddings and check whether or not we should
        # freeze embeddings from our config dict
        pretrained_embeddings = config['pretrained_embeddings'] if 'pretrained_embeddings' in config else None
        freeze_embeddings = config['freeze_embeddings'] if 'freeze_embeddings' in config else False
        if pretrained_embeddings is not None:
            # use pretrained embeddings
            self.vocab_size = pretrained_embeddings.shape[0]
            self.embedding_dim = pretrained_embeddings.shape[1]
            self.embedding = torch.nn.Embedding.from_pretrained(
                torch.from_numpy(pretrained_embeddings).float(),
                freeze=freeze_embeddings
                )
        else:
            # use randomly initialized embeddings
            assert 'vocab' in config and 'embedding_dim' in config
            self.vocab_size = config['vocab'].shape[0]
            self.embedding_dim = config['embedding_dim']
            if freeze_embeddings:
                # why would you do this?
                print(
                    'WARNING:Freezing Randomly Initialized Embeddings!!😭😭😭'
                    )
            self.embedding = torch.nn.Embedding(
                self.vocab_size,
                self.embedding_dim,
                freeze = freeze_embeddings
                )

        # store some values from the config
        self.hidden_size = config['hidden_size']
        self.lstm_unit_cnt = config['lstm_unit_cnt']

        # initialize LSTM
        self.lstm = torch.nn.LSTM(
            input_size = self.embedding_dim,
            hidden_size = self.hidden_size,
            num_layers = self.lstm_unit_cnt,

            # batch_first = T -> input dim are [batch x sentence x embedding]
            # batch_first = F -> input dim are [sentence x batch x embedding]
            batch_first = True,

            # if bidirectional is true, then the seqeunce is passed through in
            # both forward and backward directions and the results are
            # concatenated. Lookup bidirectional LSTMs for details.
            bidirectional = False
            )

        middle_nodes = int(self.hidden_size / 2)

        self.fc1 = torch.nn.Linear(in_features = self.hidden_size, out_features = middle_nodes)
        self.fc2 = torch.nn.Linear(in_features = middle_nodes, out_features = 1)
        self.relu = torch.nn.functional.relu
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, batch):
        x = batch['input_ids'].to(self.device) # lookup token ids for our inputs
        x_lengths = batch['sequence_len'] # lookup lengths of our inputs
        embed_out = self.embedding(x) # get the embeddings of the token ids

        # In pytorch, RNN's need a packed sequence object as input
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            embed_out,
            x_lengths.tolist(),
            # use if sequences are sorted by length in the batch
            enforce_sorted = False,
            batch_first = True
            )

        packed_out, (final_hidden_state, final_cell_state) = self.lstm(packed_input)

        # Inverse operation of pack_padded_sequence
        output,_ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first = True
            )

        lstm_out = output[range(len(output)), x_lengths - 1, :self.hidden_size]

        fc1_out = self.fc1(lstm_out)
        fc1_out = self.relu(fc1_out)
        fc2_out = self.fc2(fc1_out)
        final_out = self.sigmoid(fc2_out)
        return final_out

    def get_embedding_dims(self):
        return self.vocab_size, self.embedding_dim