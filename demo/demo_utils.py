import streamlit as st
import torch
import sys

if '.' not in sys.path:
    sys.path.append('.')

from models.card_name_detector.definition import TrainedDetector
from utils.translate import sentencize, CardTranslator, CTHelper


@st.cache_data
def load_translator(path: str):
    T = torch.load(path)
    return T


@st.cache_data
def load_detector():
    return TrainedDetector()


def create_card_translator(T, dict):
    D = load_detector()
    helper = CTHelper(D, dict)
    silent = True
    CT = CardTranslator(sentencize, T, 
                        preprocess=lambda x: helper.preprocess(x, silent), 
                        postprocess=lambda x: helper.postprocess(x, silent))
    return CT


from dataset.mtgcards import TestSets
from torchtext.legacy.data import Field

@st.cache_data
def load_test_data(set_code: str):
    fields = {'key': ('key', Field(tokenize=lambda x: x.split(' '))), 
              'src-rule': ('src', Field(tokenize=lambda x: x.split(' '))), 
              'trg-rule': ('trg', Field(tokenize=lambda x: x.split('\n')))}
    test_data = TestSets.load(fields, version=set_code)
    return {' '.join(data.key): {'src': ' '.join(data.src), 'trg': ' '.join(data.trg)} for data in test_data}

def reformat(str):
    num=0
    list1 = list(str)
    for i in range(len(list1)):
        if list1[i]=='<': 
            num+=1
            list1[i]=''
        if list1[i]=='>':
            num-=1
            list1[i]=''
        if u'\u4e00' <= list1[i] <= u'\u9fff':
            num=0
        if list1[i]==' ' and num==0:
            list1[i]='<br>'
    return "".join(list1)
    