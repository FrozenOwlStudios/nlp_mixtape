#!/usr/bin/env python3
# coding: utf-8

'''
A simple example of NLTK/Prolog cooperation for semantic analysis.
Example sentences "John eats an apple", "John drives the car"
'''

# ==================================================================================================
#                                             IMPORTS
# ==================================================================================================
# Future imports
from __future__ import annotations

# Python standard library
import argparse
from dataclasses import dataclass
from typing import Optional, Sequence, List

# External libraries
from nltk import CFG, ChartParser
from nltk.tokenize import word_tokenize
from nltk.tree.tree import Tree
from pyswip import Prolog # remember to do "sudo install swi-prolog" 

# ==================================================================================================
#                             CONFIGURATION AND ARGUMENT PARSING
# ==================================================================================================
@dataclass(frozen=True)
class Config:
    grammar_file: str # Path to file in which context free grammar definition resides
    reasoning_file: str # Path to prolog file with reasoning
    sentence: str # Sentence that will be processed 

    @staticmethod
    def from_args(argv: Optional[Sequence[str]] = None) -> Config:
        prsr = argparse.ArgumentParser(
            description=__doc__,
            formatter_class=argparse.RawTextHelpFormatter,
        )

        # Declare arguments
        prsr.add_argument('--grammar_file', type=str, default='grammar.txt')
        prsr.add_argument('--reasoning_file', type=str, default='reasoning.pl')
        prsr.add_argument('sentence', type=str)

        # Parsing arguments
        args: argparse.Namespace = prsr.parse_args(argv)

        # Returning config
        return Config(
            grammar_file=args.grammar_file,
            reasoning_file=args.reasoning_file,
            sentence=args.sentence
        )


# ==================================================================================================
#                                       HELPER FUNCTIONS
# ==================================================================================================

def load_grammar_from_file(filename: str)-> CFG:
    with open(filename, 'r') as grammar_file:
        grammar_str = grammar_file.read()
        return CFG.fromstring(grammar_str)
    
def prepare_semantic_query_from_sentence(sentence: str, parser: ChartParser)-> str:
    print(f'Analyzing sentence {sentence}')
    tokens = word_tokenize(sentence)
    print(f'Tokens : {tokens}')

    # For non contextual deterministic there will be only one tree
    tree: Tree = list(parser.parse(tokens))[0] 
    tree.pretty_print()
    content = {cat:val.lower() for val, cat in tree.pos()}
    print(f'Sentence content : {content}')

    query = f'can_act({content['PropN']}, {content['V']}, {content['N']}).'
    print(f'query = {query}')
    return query



# ==================================================================================================
#                                           MAIN
# ==================================================================================================

def main(argv: Optional[Sequence[str]] = None)-> None:
    cfg: Config = Config.from_args()

    grammar = load_grammar_from_file(cfg.grammar_file)
    parser = ChartParser(grammar)

    query = prepare_semantic_query_from_sentence(cfg.sentence, parser)
  
    reasoning = Prolog()
    reasoning.consult(cfg.reasoning_file)
    # For yes or no query it will return empty list for no and one with a 
    # single empty dict for yes. This is very inconvinient behavior.
    if bool(list(Prolog.query(query))):
        print(f'The sentence "{cfg.sentence}" is semantically plausible.')
    else:
        print(f'The sentence "{cfg.sentence}" is not semantically plausible.')

if __name__ == '__main__':
    main()