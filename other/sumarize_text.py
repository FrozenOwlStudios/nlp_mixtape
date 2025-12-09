#!/usr/bin/env python3
# coding: utf-8

'''
This script reads a TXT file and generates its summary consisting of given
number of most important sentences
'''

# ==================================================================================================
#                                             IMPORTS
# ==================================================================================================
# Future imports
from __future__ import annotations

# Python standard library
import argparse
import logging
import re
import sys
from dataclasses import dataclass
from enum import Enum, auto
from typing import Final, Optional, Set, List
from collections import Counter

# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag

# Other libraries
from heapq import nlargest

# ==================================================================================================
#                                      NLTK RESOURCES 
# ==================================================================================================
NLTK_RESOURCES: Final[List[str]] = [
    'stopwords',
    'punkt_tab',
    'averaged_perceptron_tagger_eng'
]

def download_nltk_resources():
    for res in NLTK_RESOURCES:
        nltk.download(res)


# ==================================================================================================
#                             CONFIGURATION AND ARGUMENT PARSING
# ==================================================================================================

def arg_string_list(txt: str) -> Tuple[str, ...]:
    entries = [entry.strip() for entry in txt.split(',') if entry.strip()]
    if not entries:
        raise argparse.ArgumentTypeError(
            'Expected a nonepty, comma separated, list.')
    return tuple(entries)

def arg_positive_int(txt: str) -> int:
    try:
        val = int(txt)
        if not val > 0:
            raise argparse.ArgumentTypeError(
                'Argument value must be greater than 0')
        return val
    except ValueError:
        raise argparse.ArgumentTypeError(
            f'Unable to convert "{txt}" to valid integer')


def arg_int(txt: str) -> int:
    try:
        val = int(txt)
        return val
    except ValueError:
        raise argparse.ArgumentTypeError(
            f'Unable to convert "{txt}" to valid integer')


def arg_verbosity_level(txt: str) -> VerbosityLevel:
    try:
        return VerbosityLevel.parse_string(txt)
    except KeyError as e:
        raise argparse.ArgumentTypeError(str(e))


@dataclass(frozen=True)
class Config:
    source_text: str  # Path to CSV with training data (inputs and outputs)
    stopwords: Tuple[str] # Additional stopwords
    lang: str # Language that will be processed

    @staticmethod
    def from_args(argv: Optional[Sequence[str]] = None) -> Config:
        prsr = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawTextHelpFormatter,
        )

        # Declare arguments
        prsr.add_argument('--source_text', required=True,
                          type=str, help='Path to training CSV')
        prsr.add_argument('--stopwords', default=[],
                          type=arg_string_list, help='Additional stopwords')
        prsr.add_argument('--lang', default='english',
                          type=str, help='Text language.')

        # Parsing arguments
        args: argparse.Namespace = prsr.parse_args(argv)

        # Returning config
        return Config(
            source_text=args.source_text,
            stopwords=args.stopwords,
            lang=args.lang
        )

# ==================================================================================================
#                                        WORD FILTER 
# ==================================================================================================
DEFAULT_ALLOWED_TAGS: Final[List[str]] =  ['NNP', 'NNPS', # Proper nouns
                                           'NN', 'NNS',   # Common nouns
                                           'JJ', 'JJR', 'JJS',    # Adjectives
                                           'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ' # Verbs
                                          ] 

class WordFilter:

    stop_words_: Set[str]
    allowed_tags_: List[str]
    working_on_list_: bool

    def __init__(self, working_on_list: bool, lang: str= 'english',
                 extra_stopwords: Optional[Iterable[str]]= None,
                 allowed_tags: Iterable[str]= DEFAULT_ALLOWED_TAGS)-> None:
        # FIXME: Add more usefull exception wrapping
        self.working_on_list_ = working_on_list
        self.stop_words_ = set(stopwords.words(lang))
        if extra_stopwords:
            for word in extra_stopwords:
                self.stop_words_.add(word)
        self.allowed_tags_ = allowed_tags

    def __call__(self, text: List[str]|str)-> List[str]|str:
        working_on_list = self.working_on_list_
        tokens = text if working_on_list else word_tokenize(text)
        print(tokens)
        filtered = [word for word in tokens if word not in self.stop_words_]
        pos_tags = pos_tag(filtered)
        filtered = [word for word, pos in pos_tags if pos in self.allowed_tags_]
        if working_on_list:
            return filtered
        return ' '.join(filtered)
    
# ==================================================================================================
#                                       NLP PIPELINE 
# ==================================================================================================

def process_text_file(source_text: str, cfg: Config)-> Dict[str, float] :
    misc_symbols = re.compile(r'[^\w\s]+') 
    allowed_words = WordFilter(True, lang=cfg.lang, extra_stopwords=cfg.stopwords)

    alphanumeric_text = misc_symbols.sub(' ', source_text)
    words = word_tokenize(alphanumeric_text)
    words = map(lambda s: s.lower(), words)
    words = allowed_words(words)
    word_counts = Counter(words)
    max_count = word_counts.most_common(1)[0][1]
    for i in word_counts.keys():
        word_counts[i]=(word_counts[i]/max_count)
    return word_counts

def get_tokenized_sentences(source_text: str, cfg: Config):
    endlines = re.compile(r'\n')
    whitespaces = re.compile(r'[\s]+')
    misc_symbols = re.compile(r'[^\w\s]+') 
    allowed_words = WordFilter(False, lang=cfg.lang, extra_stopwords=cfg.stopwords)

    sentences = sent_tokenize(source_text)
    tokenized = map(lambda s: s.lower(), sentences)
    tokenized = map(lambda s: endlines.sub(' ', s), tokenized)
    tokenized = map(lambda s: whitespaces.sub(' ', s), tokenized)
    tokenized = map(lambda s: misc_symbols.sub('', s), tokenized)
    tokenized = map(allowed_words, tokenized)
    sentences = [(s,t) for s,t in zip(sentences, tokenized)]
    return sentences

def calculate_scores(word_scores, sentences):
    sent_strength={}
    print(sentences)
    for sent in sentences:
        for word in sent[1].split(' '):
           # print(word)
            if word in word_scores.keys():
                if sent in sent_strength.keys():
                    sent_strength[sent]+=word_scores[word]
                else:
                    sent_strength[sent]=word_scores[word]
    return sent_strength


# ==================================================================================================
#                                           MAIN
# ==================================================================================================


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg: Config = Config.from_args()
    print(cfg)
    text = ''
    with open(cfg.source_text, 'r') as txt:
        text = txt.read()
    word_scores = process_text_file(text, cfg)
    print(f'Word scores size = {len(word_scores)}')
    sentences = get_tokenized_sentences(text,cfg)
    print(len(sentences))
    sentence_scores = calculate_scores(word_scores, sentences)
    print(len(sentence_scores))
    sorted_sentences = sorted(sentence_scores.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    print(len(sorted_sentences))
    top_10 = sorted_sentences[:10]
    for sentence in top_10:
        print('-------------------------------------')
        print(f'Sentence score - {sentence[1]}')
        print(sentence[0][0])
        print('-------------------------------------')
    for sentence in top_10:
        print(sentence[0][0])

if __name__ == '__main__':
    main()
