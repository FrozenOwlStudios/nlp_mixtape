#!/usr/bin/env python3
# coding: utf-8

# ==============================================================================
#                                IMPORTS
# ==============================================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet,Dict,List,Set
import faiss
from sentence_transformers import SentenceTransformer

# ==============================================================================
#                                KNOWLEDGE BASE
# ==============================================================================
@dataclass
class RelevanceScore:
    score: float
    idx: int

    def __lt__(self,other: RelevanceScore)-> bool:
        return self.score < other.score


class KnowledgeBase:

    _knowledge: Dict[str,str]
    _keyword_sets: List[str]
    _embed_model: SentenceTransformer
    _index: faiss.IndexFlatIP

    def __init__(self, knowledge: Dict[str,str],
                 sentence_transformer: SentenceTransformer|str)-> None:
        self._knowledge = knowledge
        self._keyword_sets = list(self._knowledge.keys())
        
        embed_model = None
        if isinstance(sentence_transformer, SentenceTransformer):
            embed_model = sentence_transformer
        elif isinstance(sentence_transformer, str):
            embed_model = SentenceTransformer(sentence_transformer)
        else:
            raise ValueError('Knowledge base must be given sentence '+
                             'transformer either as a SentenceTransformer '+
                             'object or string.'+ 
                             f'Received {type(sentence_transformer)} instead')
        self._embed_model = embed_model

        embeddings = self._embed_model.encode(
            self._keyword_sets,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings) # pyright: ignore[reportCallIssue] (error in docs)

    def get_available_keywords(self)-> Set[str]:
        all_keywords: Set[str] = set()
        for keywords in self._knowledge.keys():
            all_keywords.update(keywords.split())
        return all_keywords
    
    def get_relevance_scores(self, query:str, k=5, min_score=0.3)-> List[RelevanceScore]:
        query_embed = self._embed_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores, indices = self._index.search(query_embed, k=k) # pyright: ignore[reportCallIssue] (error in docs)
        print(scores, indices)
        rev_scores = [RelevanceScore(score,idx) for score, idx in zip(scores[0],indices[0]) if score>min_score]
        rev_scores.sort()
        return rev_scores


    @staticmethod
    def load_knowledge_file(file_path: str,
                            sentence_transformer: SentenceTransformer|str)-> KnowledgeBase:
        knowledge: Dict[FrozenSet[str],List[str]] = {}
        with open(file_path, "r", encoding="utf-8") as knowledge_file:
            keywords: FrozenSet[str] = frozenset()
            information_content: List[str] = []
            for line in knowledge_file:
                line = line.strip()
                
                # Ignore empty lines
                if len(line) == 0:
                    continue
                
                if line.startswith("[") and line.endswith("]"):
                    if len(information_content) != 0:
                        content = '\n'.join(information_content)
                        if keywords in knowledge:
                            knowledge[keywords].append(content)
                        else:
                            knowledge[keywords] = [content]
                    keywords = frozenset(line[1:-1].lower().split())
                    information_content = []
                else:
                    information_content.append(line)
            if len(information_content) != 0:
                content = '\n'.join(information_content)
                if keywords in knowledge:
                    knowledge[keywords].append(content)
                else:
                    knowledge[keywords] = [content]
        return KnowledgeBase({' '.join(keywords):'\n'.join(content)
                              for keywords, content in knowledge.items()},
                              sentence_transformer)

# ==============================================================================
#                                MAIN
# ==============================================================================

def main()-> None:
    knowledge_base = KnowledgeBase.load_knowledge_file('example_knowledge.txt',
                                                       "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    print(knowledge_base.get_available_keywords())
    query = ''
    while query.lower() != 'exit':
        query = input('Query => ')
        relevance = knowledge_base.get_relevance_scores(query)
        print(relevance)

if __name__ == '__main__':
    main()