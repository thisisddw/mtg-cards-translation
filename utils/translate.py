class Translator:
    def __init__(self, src_field, trg_field, model, device, beam_search) -> None:
        """
        beam_search: (sentence, src_field, trg_field, model, device, max_len, beam_size)->tuple[tokens, probs, ...]
        """
        self.src_field = src_field
        self.trg_field = trg_field
        self.model = model
        self.device = device
        self.beam_search = beam_search
    
    def translate(self, text: str, max_len: int = 50, beam_size: int = 3):
        """
        Returns a tuple[list[list[str]], list[float]], where the first
        list is topk translations, the second is their corresponding probabilities.
        """
        return self.beam_search(text, self.src_field, self.trg_field, self.model, self.device,
                                max_len=max_len, beam_size=beam_size)[:2]
        
    def translate_with_att(self, text: str, max_len: int = 50, beam_size: int = 3):
        """
        Comparing to the translate method, this one returns a list of attention weights additionally.
        """
        return self.beam_search(text, self.src_field, self.trg_field, self.model, self.device,
                                max_len=max_len, beam_size=beam_size)[:3]
    

def sentencize(text: str):
    ignore = {' ', '(', ')', '\n'}
    while len(text) and text[0] in ignore:
        text = text[1:]
    if len(text) == 0:
        return []
    
    r = 0
    delims = {'.', '\n', '('}
    ignore = False
    while r < len(text):
        if text[r] == '\"':
            ignore = not ignore
        if not ignore and text[r] in delims:
            break
        r += 1
    
    if r < len(text) and text[r] == '.':
        return [text[:r + 1]] + sentencize(text[r + 1:])
    return [text[:r]] + sentencize(text[r:])


class CardTranslator:
    def __init__(self, sentencize, sent_translator, preprocess=None, postprocess=None) -> None:
        self.sentencize = sentencize
        self.sent_translator = sent_translator
        self.preprocess = preprocess
        self.postprocess = postprocess
    
    def translate(self, text: str)->str:
        sents = self.sentencize(text)
        result = []
        for sent in sents:
            if self.preprocess:
                sent = self.preprocess(sent)
            sent, _ = self.sent_translator.translate(sent)
            sent = ''.join(sent[0][:-1])
            if self.postprocess:
                sent = self.postprocess(sent)
            result.append(sent)
        return ' '.join(result)


import re
class CTHelper:
    def __init__(self, name_detector, dictionary={}) -> None:
        self.D = name_detector
        self.dictionary = dictionary
    
    def preprocess(self, x:str, silent:bool=True):
        self.tag2str = {}
        x = self.D.annotate(x).removeprefix(' ') # x become lowercase after go through detector
        m = re.search('<[^0-9>]+>', x)
        id = 0
        while m:
            l, r = m.span()
            tag = '<' + str(id) + '>'
            self.tag2str[tag] = x[l:r]
            x = x[:l] + tag + x[r:]
            id += 1
            m = re.search('<[^0-9>]+>', x)

        for s in self.dictionary.keys():
            m = re.search(s, x)
            if m:
                tag = '<' + str(id) + '>'
                self.tag2str[tag] = s
                x = x.replace(s, tag)
                id += 1
        if not silent:
            print(f'[after preprocess]:{x}')
        return x

    def postprocess(self, x:str, silent:bool=True):
        if not silent:
            print(f'[before postprocess]:{x}')
        for tag, s in self.tag2str.items():
            x = x.replace(tag, self.dictionary[s] if s in self.dictionary else s)
        return x