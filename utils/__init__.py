import torch

import math
import time
import os
import random

from tqdm import tqdm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_loop(model, optimizer, criterion, train, evaluate,
                train_iterator, valid_iterator, N_EPOCHS=10, CLIP=1, 
               save_path='', file_name='model.pt', load_before_train=True):

    path = save_path + file_name
    print(f'model will be saved to {path}')

    best_valid_loss = float('inf')

    if load_before_train and os.path.isfile(path):
        model.load_state_dict(torch.load(path, map_location=torch.device(model.device)))
        best_valid_loss = evaluate(model, valid_iterator, criterion)
        print(f'load model parameters from {path}\nVal. Loss: {best_valid_loss:.3f} |  Val. PPL: {math.exp(best_valid_loss):7.3f}')

    for epoch in range(N_EPOCHS):
        
        start_time = time.time()
        
        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), path)
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


def tok_cat(toks: list, sep: str = '')->str:
    s = ''
    for t in toks:
        s += t + sep
    return s

def show_samples(dataset, translator, n=5, beam_size=10):
    for s in random.sample(list(dataset), n):
        toks,probs = translator.translate(s.src, max_len = 50, beam_size=beam_size)
        tmp = tok_cat(s.src, ' ')
        print(f'src: [{tmp}] trg = [{tok_cat(s.trg)}]')
        for i in range(3):
            print(tok_cat(toks[i]), f'\t[probability: {probs[i]:.5f}]')
        print('')


from torchtext.data.metrics import bleu_score

def calculate_bleu(data, translate):
    
    trgs = []
    pred_trgs = []
    
    for datum in tqdm(data):
        
        src = vars(datum)['src']
        trg = vars(datum)['trg']
        
        pred_trg = translate(src)
        
        #cut off <eos> token
        pred_trg = pred_trg[:-1]
        
        pred_trgs.append(pred_trg)
        trgs.append([trg])
        
    return bleu_score(pred_trgs, trgs)
