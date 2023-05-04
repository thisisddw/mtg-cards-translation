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
                                max_len=50, beam_size=beam_size)[:2]
        
    def translate_with_att(self, text: str, max_len: int = 50, beam_size: int = 3):
        """
        Compare to translate, returns a list of attention weights additionally.
        """
        return self.beam_search(text, self.src_field, self.trg_field, self.model, self.device,
                                max_len=50, beam_size=beam_size)[:3]