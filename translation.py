"""Local German-to-English translation with the Helsinki-NLP Opus-MT model."""

import re


MODEL_NAME = "Helsinki-NLP/opus-mt-de-en"


class Translator:
    def __init__(self, device="cpu", english_fixes=None):
        import torch
        from transformers import MarianMTModel, MarianTokenizer

        self.torch = torch
        self.device = device
        self.english_fixes = english_fixes or {}
        self.tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
        self.model = MarianMTModel.from_pretrained(MODEL_NAME).to(device).eval()

    def translate(self, german_text):
        inputs = self.tokenizer(
            [german_text], return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, num_beams=1, max_new_tokens=256)
        english_text = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        for wrong, right in self.english_fixes.items():
            english_text = re.sub(
                rf"\b{re.escape(wrong)}\b", right, english_text, flags=re.IGNORECASE
            )
        return english_text