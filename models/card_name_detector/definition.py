import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchtext.legacy.data import Field, TabularDataset, BucketIterator

import random
import math
import time
import os
import re
import spacy
from typing import Callable

from tqdm import tqdm


class Detector(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, output_dim, num_layers, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.GRU(emb_dim, hid_dim, num_layers = num_layers, bidirectional = True)

        self.fc_out = nn.Linear(hid_dim * 2 + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_len):
        
        #src = [src len, batch size]
        #src_len = [batch size]
        
        embedded = self.dropout(self.embedding(src))
        
        #embedded = [src len, batch size, emb dim]
                
        #need to explicitly put lengths on cpu!
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.to('cpu'))
                
        packed_outputs, hidden = self.rnn(packed_embedded)
                                 
        #packed_outputs is a packed sequence containing all hidden states
        #hidden is now from the final non-padded element in the batch
            
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs) 
            
        #outputs is now a non-packed sequence, all hidden states obtained
        #  when the input is a pad token are all zeros
            
        #outputs = [src len, batch size, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]
        
        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer
        
        outputs = self.fc_out(torch.cat((outputs, embedded), 2))
        # outputs = self.fc_out(outputs)
        
        #outputs = [src len, batch size, output_dim]
        
        return outputs
    

from utils.preprocess import tokenize_en, tokenize_zh
import json

def detect_card_name(sentence: str, src_field: Field, trg_field: Field, model, device, **kwargs):
    model.eval()

    with torch.no_grad():
        tokens = src_field.preprocess(sentence)
        print(tokens)
        input, len = src_field.process([tokens])
        logits = model(input.to(device), len).squeeze(dim=1)
        probs = F.softmax(logits, dim=1)[:,trg_field.vocab.stoi['1']]
        # id_list = logits.argmax(dim=1)
        # output = [trg_field.vocab.itos[x] for x in id_list]

    return tokens, [x.item() for x in list(probs)[1:-1]]


class TrainedDetector:
    """
    A trained model for card name detection in a rule-text sentence.
    """
    def __init__(self) -> None:
        self.SRC = Field(tokenize = tokenize_en, 
                        init_token = '<sos>', 
                        eos_token = '<eos>', 
                        lower = True, 
                        include_lengths=True)
        self.TRG = Field(init_token = '0', 
                        eos_token = '0', 
                        lower = True)
        
        path = os.path.dirname(os.path.abspath(__file__))
        print(f'path: {path}')

        self.SRC.vocab = torch.load(path + '/src_vocab.pt')
        self.TRG.vocab = torch.load(path + '/trg_vocab.pt')
        
        INPUT_DIM = len(self.SRC.vocab)
        OUTPUT_DIM = len(self.TRG.vocab)
        EMB_DIM = 256
        HID_DIM = 512
        DROPOUT = 0.5
        NUM_LAYERS = 2
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = Detector(INPUT_DIM, EMB_DIM, HID_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT).to(self.device)
        self.model.load_state_dict(torch.load(path + '/cn-mask-model.pt', map_location=torch.device(self.device)))
    
    def detect(self, sentence: str):
        """
        args:
            sentence: A rule-text sentence to detect.
        return:
            list[tuple(str, float)], a list of token-probability pairs.
        """
        toks, probs = detect_card_name(sentence, self.SRC, self.TRG, self.model, self.device)
        ret = [(t,p) for t,p in zip(toks, probs)]
        # print(ret)
        return ret
    
    def annotate(self, sentence: str, threshold: float = 0.5)->str:
        tokens = self.detect(sentence)
        ret = ''
        state = 0
        for tok, p in tokens:
            if p >= threshold and state == 0:
                state = 1
                ret += ' <'
            elif p < threshold and state == 1:
                state = 0
                ret += '> '
            else:
                ret += ' '
            ret += tok            
        return ret

    