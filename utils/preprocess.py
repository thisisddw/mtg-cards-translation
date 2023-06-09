from torchtext.legacy.data import Field, TabularDataset, BucketIterator
import re
import spacy
from typing import Callable

spacy_en = spacy.load('en_core_web_sm')
spacy_zh = spacy.load("zh_core_web_sm")

special_symbol_exp = '\{[^{}]*\}|[\dX/+-]|<cn>|<[^>]*>'
def tokenize_wraper(text: str, tokenize: Callable[[str], list[str]]):
    """
    Tokenizes text to a list of tokens using the arg 'tokenize', except split substrings 
    matching special_symbol_exp into single characters
    """
    tokens = []
    while len(text):
        m = re.search(special_symbol_exp, text)
        if m is None:
            tokens = tokens + [tok.text for tok in tokenize(text)]
            break
        else:
            l,r = m.span()
            tokens = tokens + [tok.text for tok in tokenize(text[:l])]
            if text[l:r] == '<cn>':
                tokens.append('<cn>')
            else:
                for i in range(l, r):
                    tokens.append(text[i])
            text = text[r:]
            while len(text) and text[0] == ' ':
                text = text[1:]
    ret = []
    for token in tokens:
        if len(token) > 3 and token[0:3] == 'non':
            ret.append('non')
            ret.append(token[3:])
        else:
            ret.append(token)
    return ret

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings (tokens) and reverses it
    """
    return tokenize_wraper(text, spacy_en.tokenizer)

def tokenize_zh(text):
    """
    Tokenizes Chinese text from a string into a list of strings (tokens)
    """ 
    return tokenize_wraper(text, spacy_zh.tokenizer)

def letter_level_tokenizer_en(text):
    """
    Tokenizes English text from a string into a list of strings containing on a letter
    """
    return [letter for letter in text]

def fields_for_rule_text(include_lengths = True, batch_first = False):
    """
    return a tupe (SRC, TRG)
    """
    SRC = Field(tokenize = tokenize_en, 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True, 
                include_lengths=include_lengths,
                batch_first=batch_first)
    TRG = Field(tokenize = tokenize_zh, 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True,
                batch_first=batch_first)
    return SRC, TRG
