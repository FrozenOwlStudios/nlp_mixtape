#!/usr/bin/env python3
# coding: utf-8

"""
This script analyzes a shift in focus and sentiment through the book.
It splits contents using chapter marks that have to be given as a regular
expression. Any content before first chapter marker is ignored.
Conclusions as well as some details of the process are written as an pdf
file.
"""

# ==============================================================================
#                                  INCLUDES
# ==============================================================================

# Better support for annotations (future imports must be first)
from __future__ import annotations

# Basic python imports
import sys
from pathlib import Path
from argparse import ArgumentParser, ArgumentTypeError
from dataclasses import dataclass
from typing import NoReturn, Optional, Sequence, List, Dict, TextIO, Set, Counter, Tuple
from io import StringIO

# General utility libraries
import re
import regex as rgx

# Visualization tools
from fpdf import FPDF
import matplotlib.pyplot as plt  # pip install PyQt5
from wordcloud import WordCloud

# NLP related stuff
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer


# ==============================================================================
#                            HELPFULL METHODS
# ==============================================================================
def panic(txt: str) -> NoReturn:
    print(txt, file=sys.stderr, flush=True)
    sys.exit(1)


# ==============================================================================
#                              CONFIGURATION
# ==============================================================================
def arg_existing_path(txt: str) -> Path:
    path = Path(txt)
    if not path.exists() or not path.is_file():
        err_msg = f"Unable to fine file at path {path.absolute()}"
        raise ArgumentTypeError(err_msg)
    return path


def arg_nonexistent_path(txt: str) -> Path:
    path = Path(txt)
    if path.exists():
        err_msg = f"Path {path.absolute()} already exists"
        raise ArgumentTypeError(err_msg)
    return path


def arg_path(txt: str) -> Path:
    return Path(txt)


def arg_regex(txt: str) -> rgx.Pattern:
    try:
        regex = rgx.compile(txt)
        return regex
    except Exception:
        msg = f'Patten "{txt}" is not a valid POSIX regex.'
        raise ArgumentTypeError(msg)


@dataclass(frozen=True)
class Config:
    # Required arguments
    text_path: Path  # Path to source text
    out_path: Path  # Path where output will be stored
    chapter_marker: str  # POSIX compatibile regular expression
    # indicating start of chapter

    # Optional arguments
    metadata_path: Path | None  # Path to txt file with metadata, if provided
    # resulting raport will be more verbose.
    team: str | None  # List of people in research team, if provided this
    # information will be added to raport.

    @staticmethod
    def from_args(
        desc: Optional[str] = None, argv: Optional[Sequence[str]] = None
    ) -> Config:
        prsr = ArgumentParser(description=desc)

        prsr.add_argument(
            "-t",
            "--text_path",
            type=arg_existing_path,
            required=True,
            help="Path to source text.",
        )
        prsr.add_argument(
            "-o",
            "--out_path",
            type=arg_path,
            required=True,
            help="Path where output will be stored.",
        )
        prsr.add_argument(
            "-f",
            "--force",
            action="store_true",
            help="Path where output will be stored.",
        )
        prsr.add_argument(
            "-c",
            "--chapter_marker",
            type=str,
            required=True,
            help="POSIX compatibile regular expression indicating start of chapter.",
        )
        prsr.add_argument(
            "-m",
            "--metadata_path",
            type=arg_existing_path,
            default=None,
            help="Path to txt file with metadata, if provided"
            "resulting raport will be more verbose.",
        )
        prsr.add_argument(
            "--team",
            type=str,
            default=None,
            help="List of people in research team, if provided "
            "this information will be added to report.",
        )

        args = prsr.parse_args(argv)

        return Config(
            text_path=args.text_path,
            out_path=args.out_path,
            chapter_marker=args.chapter_marker,
            metadata_path=args.metadata_path,
            team=args.team,
        )


# ==================================================================================================
#                                     TEXT FILE PROCESSING
# ==================================================================================================

NUMERAL_REGEX = r"([0-9]+)|([IVXLCDM]+)"
SEPARATOR_REGEX = r"[\s:,\.]+"


@dataclass
class Chapter:
    idx: int
    title: str
    content: str


class Book:
    title: str
    _by_order: List[Chapter]
    _by_title: Dict[str, Chapter]

    def __init__(self, title: str, chapters: List[Chapter]) -> None:
        self.title = title
        self._by_order = sorted(chapters, key=lambda chapter: chapter.idx)
        self._by_title = {}
        for chapter in chapters:
            if chapter.title in self._by_title:
                err_msg = f'Chapter with name "{chapter.title}" appears multiple times in book "{title}".'
                raise ValueError(err_msg)
            self._by_title[chapter.title] = chapter

    def get_chapter(self, chapter_id: int | str) -> Chapter:
        if isinstance(chapter_id, int):
            if chapter_id < 0 or chapter_id >= len(self._by_order):
                err = (
                    f"Requested chapter with index {chapter_id} while value"
                    f' from 0 to {len(self._by_order)} is required for book "{self.title}".'
                )
                raise KeyError(err)
            return self._by_order[chapter_id]
        try:
            return self._by_title[chapter_id]
        except KeyError:
            err = (
                f'Chapter with name "{chapter_id}" do not exist in book "{self.title}".'
            )
            raise KeyError(err)

    def get_chapter_list(self) -> List[str]:
        return [chapter.title for chapter in self._by_order]

    def __iter__(self) -> BookIterator:
        return BookIterator(self)


class BookIterator:
    book: Book
    chapter_ptr: int

    def __init__(self, book: Book) -> None:
        self.book = book
        self.chapter_ptr = 0

    def __next__(self) -> Chapter:
        try:
            chapter = self.book.get_chapter(self.chapter_ptr)
            self.chapter_ptr += 1
            return chapter
        except KeyError:
            raise StopIteration()


class ChaptersFactory:
    _chapter_marker: str
    _remove_numbering: bool

    _txt_file: TextIO
    _line: str

    def __init__(self, chapter_marker: str, remove_numbering: bool):
        self._chapter_marker = chapter_marker
        self._remove_numbering = remove_numbering
        self._txt_file = StringIO()
        self._line = ""

    def parse_text_file(self, title: str, path: str) -> Book:
        self._txt_file = open(path, "r")
        self._readline()

        chapters: List[Chapter] = []
        self._skip_header()
        idx = 1
        while self._line:
            ch_title = self._extract_chapter_name()
            content = self._extract_content()
            chapters.append(Chapter(idx, ch_title, content))
            idx += 1

        return Book(title, chapters)

    def _readline(self) -> None:
        raw_line = self._txt_file.readline()
        line = raw_line.strip()
        if len(raw_line) > 0 and len(line) == 0:
            self._line = raw_line
            return
        self._line = line

    def _check_for_EOF(self) -> None:
        if not self._line:
            raise ValueError("Reached EOF.")

    def _extract_content(self) -> str:
        content = []
        while self._line and self._chapter_marker not in self._line:
            content.append(self._line)
            self._readline()
        return "\n".join(content)

    def _extract_chapter_name(self):
        line_elements = re.split(SEPARATOR_REGEX, self._line)
        try:
            line_elements.remove(self._chapter_marker)
        except ValueError:
            print(
                f"Unable to delete {self._chapter_marker} from string {line_elements}"
            )
            sys.exit()
        chapter_name = []
        numeral = re.compile(NUMERAL_REGEX)
        skip_disabled = False
        while not chapter_name:
            self._check_for_EOF()
            for element in line_elements:
                if skip_disabled or not numeral.fullmatch(element):
                    chapter_name.append(element)
                    skip_disabled = True
            if not chapter_name:
                self._readline()
                line_elements = re.split(SEPARATOR_REGEX, self._line)
                line_elements = [e for e in line_elements if e != ""]

        # If chapter name was in same line as its keyword we need to skip one line
        if self._chapter_marker in self._line:
            self._readline()
        return " ".join(line_elements)

    def _skip_header(self) -> None:
        header_size = 0
        while self._line and self._chapter_marker not in self._line:
            header_size += 1
            self._readline()
        self._check_for_EOF()
        print(f"Ignored {header_size} of header lines.")


# ==============================================================================
#                            TEXT ANALYSIS
# ==============================================================================
class Topics:
    topics: Set[str]
    threshold: int  # How many most frequent words we consider topics

    def __init__(self, threshold: int) -> None:
        self.topics = set()
        self.threshold = threshold

    def add_predefined(self, new_topics: Sequence[str]) -> None:
        self.topics.update(new_topics)

    def add_from_distribution(self, distribution: Counter[str]) -> None:
        new_topics = distribution.most_common(self.threshold)
        new_topics = [topic for topic, _ in new_topics]
        self.topics.update(new_topics)

    def __iter__(self):
        return self.topics.__iter__()


@dataclass
class TopicInfo:
    frequency: int
    normalized_frequency: float
    sentiment: float


@dataclass
class ChapterInfo:
    sentiment: float
    topics: Dict[str, TopicInfo]


class BookAnalyzer:
    book: Book
    tokens: Dict[str, List[str]]
    sentiment_analyzer: SentimentIntensityAnalyzer
    MIN_WORD_LENGTH = 3

    def __init__(self, book: Book, extra_stop_words: List[str]) -> None:
        self.book = book
        self.tokens = {}
        stop_words = set(stopwords.words("english")).union(extra_stop_words)
        for chapter in book:
            self.tokens[chapter.title] = self._clean_tokens(chapter.content, stop_words)
        self._sentiment_analyzer = SentimentIntensityAnalyzer()

    def run_analysis(
        self, topic_threshold: int
    ) -> Tuple[Topics, Dict[str, ChapterInfo]]:
        chapter_info: Dict[str, ChapterInfo] = {}
        word_distributions = {
            chapter: Counter(tokens) for chapter, tokens in self.tokens.items()
        }
        topics = Topics(topic_threshold)
        for _, dist in word_distributions.items():
            topics.add_from_distribution(dist)
        for chapter in self.book:
            chapter_info[chapter.title] = self.get_chapter_info(
                chapter.content,
                self.tokens[chapter.title],
                topics,
                word_distributions[chapter.title],
            )
        return topics, chapter_info

    def get_chapter_info(
        self, txt: str, tokens: List[str], topics: Topics, word_distribution: Counter
    ) -> ChapterInfo:
        full_sentiment = self._sentiment_score(txt)
        topic_info: Dict[str, TopicInfo] = {}

        for topic in topics:
            frequency = word_distribution[topic]
            normalized_frequency = frequency / max(len(tokens), 1)

            topic_sentences = [
                sentence
                for sentence in sent_tokenize(txt)
                if topic.lower() in sentence.lower()
            ]

            if topic_sentences:
                topic_sentiment = sum(
                    self._sentiment_score(sentence) for sentence in topic_sentences
                ) / len(topic_sentences)
            else:
                topic_sentiment = 0.0

            topic_info[topic] = TopicInfo(
                frequency=frequency,
                normalized_frequency=normalized_frequency,
                sentiment=topic_sentiment,
            )
        return ChapterInfo(sentiment=full_sentiment, topics=topic_info)

    def _sentiment_score(self, txt: str) -> float:
        return self._sentiment_analyzer.polarity_scores(txt)["compound"]

    @staticmethod
    def _clean_tokens(txt: str, stop_words: Set[str]) -> List[str]:

        tokens = word_tokenize(txt.lower())
        tokens = [
            t
            for t in tokens
            if t.isalpha()
            and t not in stop_words
            and len(t) >= BookAnalyzer.MIN_WORD_LENGTH
        ]

        return tokens


def select_important_topics(
    topics: Topics, analysis: Dict[str, ChapterInfo], top_n: int
) -> List[str]:
    max_norm_freq: List[Tuple[str, float]] = []
    for topic in topics:
        max_nfrq = 0.0
        for _, chapter_stats in analysis.items():
            if chapter_stats.topics[topic].normalized_frequency > max_nfrq:
                max_nfrq = chapter_stats.topics[topic].normalized_frequency
        max_norm_freq.append((topic, max_nfrq))
    max_norm_freq = sorted(max_norm_freq, key=lambda x: x[1], reverse=True)
    return [mnf[0] for mnf in max_norm_freq[:top_n]]


# ==============================================================================
#                                 PLOTTING
# ==============================================================================
def create_wordcloud(counter: Counter, output_path: str):
    wc = WordCloud(
        width=1200, height=700, background_color="white"
    ).generate_from_frequencies(counter)

    plt.figure(figsize=(12, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_topic_frequencies(
    book: Book,
    analysis: Dict[str, ChapterInfo],
    topics: Topics,
    output_path: str,
    important_topics: Optional[List[str]],
) -> None:
    plt.figure(figsize=(12, 7))
    chapters = [chapter.title for chapter in book]

    for topic in topics:
        if important_topics and topic not in important_topics:
            continue
        freq = []
        for chapter in book:
            freq.append(analysis[chapter.title].topics[topic].normalized_frequency)

        plt.plot(chapters, freq, marker="o", label=topic)

    plt.title("Topic Frequency Drift Across Files")
    plt.xlabel("Chapter")
    plt.ylabel("Normalized Frequency")
    plt.xticks(rotation=70)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_sentiment_drift( book: Book,
    analysis: Dict[str, ChapterInfo],
    topics: Topics,
    output_path: str,
    important_topics: Optional[List[str]]):
    plt.figure(figsize=(12, 7))
    chapters = [chapter.title for chapter in book]

    ch_sent = []
    for chapter in book:
        ch_sent.append(analysis[chapter.title].sentiment)
    plt.plot(chapters, ch_sent, marker="o", label="overall sentiment")

    for topic in topics:
        if important_topics and topic not in important_topics:
            continue
        sent = []
        for chapter in book:
            sent.append(analysis[chapter.title].topics[topic].sentiment)

        plt.plot(chapters, sent, marker="o", label=topic)
    

    plt.axhline(0, linestyle="--")
    plt.title("Sentiment Drift Across Files")
    plt.xlabel("Chapter")
    plt.xticks(rotation=70)
    plt.ylabel("Sentiment Polarity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# ==============================================================================
#                            PDF WRITING WRAPPER
# ==============================================================================
class PdfWriter:
    pdf: FPDF

    def __init___(self) -> None:
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)

    def prepare_title_page(self):
        self.pdf.add_page()
        self.pdf.set_font("Arial", "B", 18)
        self.pdf.cell(0, 10, "Topic and Sentiment Drift Report for ", ln=True)


# =============================================================================
#                                MAIN
# =============================================================================
def main(argv: Optional[Sequence[str]] = None) -> None:
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("stopwords")
    nltk.download("vader_lexicon")
    cfg = Config.from_args(__doc__, argv)
    print(cfg)
    chf = ChaptersFactory(cfg.chapter_marker, False)
    book = chf.parse_text_file("BOOK TITLE", str(cfg.text_path.absolute()))
    for chapter in book:
        print(chapter.title)
    analyzer = BookAnalyzer(book, ["said"])
    topics, analysis = analyzer.run_analysis(5)
    for chapter, data in analysis.items():
        print(f"{chapter} = {data.sentiment}")
    plot_topic_frequencies(
        book,
        analysis,
        topics,
        f"{cfg.out_path}/topics.png",
        select_important_topics(topics, analysis, 5),
    )

    plot_sentiment_drift(
        book,
        analysis,
        topics,
        f"{cfg.out_path}/sentiment.png",
        select_important_topics(topics, analysis, 5),
    )

if __name__ == "__main__":
    main()
