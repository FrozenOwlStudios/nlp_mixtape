#!/usr/bin/env python3
# coding: utf-8

# ==============================================================================
#                                IMPORTS
# ==============================================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet,Dict,List,Set

import faiss
import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline


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
        rev_scores = [RelevanceScore(score,idx) for score, idx in zip(scores[0],indices[0]) if score>min_score]
        rev_scores.sort()
        return rev_scores
    
    def get_relevant_context(self,query:str, k=5, min_score=0.3)-> str:
        relevance_scores = self.get_relevance_scores(query,k,min_score)
        context = []
        for score in relevance_scores:
            context.append(self._knowledge[self._keyword_sets[score.idx]])
        if len(context) == 0:
            return 'No information found'
        return '\n'.join(context)


    @staticmethod
    def load_knowledge_file(file_path: str,
                            sentence_transformer: SentenceTransformer|str,
                            use_content_as_keywords = False)-> KnowledgeBase:
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
                        if use_content_as_keywords:
                            keywords = frozenset(content.split())
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
                if use_content_as_keywords:
                    keywords = frozenset(content.split())
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

CHAT_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
KNOWLEGE_FILE_PATH = 'example_knowledge.txt'
SENTENCE_TRANSFORMER_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

def prepare_prompt(user_message: str, context: str = "") -> str:
    system_message = (
        "You are helpfull asistant.\n"
    )

    context_block = ""
    if context:
        context_block = (
            "Ruleset:\n"
            f"{context.strip()}\n\n"
        )

    # Simple chat-style format
    # TinyLlama understands generic chat prompts reasonably well.
    prompt = (
        f"{system_message}"
        f"User: {user_message.strip()}\n"
        f"Answer using only following context {context_block}"
        f"Bot:"
    )
    return prompt



def chat_once(user_message: str, extra_context: str = "") -> str:
    prompt = prepare_prompt(user_message, extra_context)
    print(prompt)
    output = pipe(prompt)[0]["generated_text"]

    if "Assistant:" in output:
        answer = output.split("Assistant:", 1)[1]
    elif "Bot:" in output:
        answer = output.split("Bot:", 1)[1]
    else:
        # Fallback if formatting changes
        answer = "I do not know."

    return answer.strip()


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        CHAT_MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    pipe = TextGenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
    )

    knowledge_base = KnowledgeBase.load_knowledge_file(KNOWLEGE_FILE_PATH,
                                                       SENTENCE_TRANSFORMER_NAME,
                                                       use_content_as_keywords=True)

    while True:
        query = input("You: ")
        if query.lower().strip() in {"quit", "exit"}:
            break

        # Example: manually supplied context per turn
        context = knowledge_base.get_relevant_context(query)
        reply = chat_once(query, extra_context=context)
        print('===============================================================')
        print("Answer:", reply)
