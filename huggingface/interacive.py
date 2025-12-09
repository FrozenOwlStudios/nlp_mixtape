#!/usr/bin/env python3
# coding: utf-8

# This code uses a huggingface library and resources to run a simple interactive
# sentiment analisys for Polish language. This can be done in even less lines
# of code with pipeline but I wanted to have an example of more step by step
# approach.

# ==============================================================================
#                                IMPORTS
# ==============================================================================
import torch
from datasets import load_dataset
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ==============================================================================
#                                 MAIN
# ==============================================================================

PL_LABEL_NAMES= {0: 'negatywny',
                 1: 'neutralny',
                 2: 'pozytywny'}
CHECKPOINT = 'eevvgg/PaReS-sentimenTw-political-PL'

def main():
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT)
    
    print('Wpisz wypowiedź by otrzymać ocenę jej wydźwięku albo "STOP" '
          'żeby zakończyć działanie programu.')
    while True:
        sentence = input('Wypowiedź ->')
        if sentence == 'STOP':
            break
        tokens = tokenizer(sentence, return_tensors="pt")
        output = model(**tokens)
        predictions = torch.nn.functional.softmax(output.logits, dim=-1)
        classification = predictions.argmax(1)
        print(f'Twoja wypowiedź ma {PL_LABEL_NAMES[int(classification.item())]} charakter.')    

if __name__ == '__main__':
    main()
