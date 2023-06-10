from .definition import *
import os
import json


def create_model(INPUT_DIM, OUTPUT_DIM, SRC_PAD_IDX, TRG_PAD_IDX, device,
                 config = 'default'):
    
    if isinstance(config, str):
        path = os.path.dirname(os.path.abspath(__file__)) + f'/configs/{config}.json'
        print(f'Load parameters from {path}')
        with open(path, 'r') as f:
            config = json.load(f)

    # HID_DIM = 256
    # ENC_LAYERS = 3
    # DEC_LAYERS = 3
    # ENC_HEADS = 8
    # DEC_HEADS = 8
    # ENC_PF_DIM = 512
    # DEC_PF_DIM = 512
    # ENC_DROPOUT = 0.1
    # DEC_DROPOUT = 0.1

    HID_DIM = config['HID_DIM']
    ENC_LAYERS = config['ENC_LAYERS']
    DEC_LAYERS = config['DEC_LAYERS']
    ENC_HEADS = config['ENC_HEADS']
    DEC_HEADS = config['DEC_HEADS']
    ENC_PF_DIM = config['ENC_PF_DIM']
    DEC_PF_DIM = config['DEC_PF_DIM']
    ENC_DROPOUT = config['ENC_DROPOUT']
    DEC_DROPOUT = config['DEC_DROPOUT']

    print(f'Parameters: {config}')

    enc = Encoder(INPUT_DIM, 
                HID_DIM, 
                ENC_LAYERS, 
                ENC_HEADS, 
                ENC_PF_DIM, 
                ENC_DROPOUT, 
                device)

    dec = Decoder(OUTPUT_DIM, 
                HID_DIM, 
                DEC_LAYERS, 
                DEC_HEADS, 
                DEC_PF_DIM, 
                DEC_DROPOUT, 
                device)
    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

    return model