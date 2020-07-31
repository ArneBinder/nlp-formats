# This code is an adapted version of https://github.com/huggingface/nlp/blob/cab7ff378a37cf212730b3ce039b5e2bd95c9454/datasets/germeval_14/germeval_14.py


from __future__ import absolute_import, division, print_function

from abc import ABC, abstractmethod
import csv
import glob
import logging
import os
from typing import List
from pathlib import Path

import nlp


@dataclass
class ConllConfig(nlp.BuilderConfig):
    """BuilderConfig for BRAT."""
  
    columns: List[str] = ["id", "source", "tokens", "labels", "nested-labels"]

class AbstractConll(nlp.GeneratorBasedBuilder, ABC):
    """GermEval 2014 NER Shared Task dataset."""


    def _info(self):
        return nlp.DatasetInfo(
            features=nlp.Features(
                {
                    self.config.columns[0]: nlp.Value("string"),
                    self.config.columns[1]: nlp.Value("string"),
                    self.config.columns[2]: nlp.Sequence(nlp.Value("string")),
                    self.config.columns[3]: nlp.Sequence(nlp.Value("string")),
                    self.config.columns[4]: nlp.Sequence(nlp.Value("string")),
                }
            )
        )

    def _generate_examples(self, filepath):
        logging.info("⏳ Generating examples from = %s", filepath)
        with open(filepath) as f:
            data = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            current_source = ""
            current_tokens = []
            current_labels = []
            current_nested_labels = []
            sentence_counter = 0
            for row in data:
                if row:
                    if row[0] == "#":
                        current_source = " ".join(row[1:])
                        continue
                    id_, token, label, nested_label = row[:4]
                    current_tokens.append(token)
                    current_labels.append(label)
                    current_nested_labels.append(nested_label)
                else:
                    # New sentence
                    if not current_tokens:
                        # Consecutive empty lines will cause empty sentences
                        continue
                    assert len(current_tokens) == len(current_labels), "💔 between len of tokens & labels"
                    assert len(current_labels) == len(current_nested_labels), "💔 between len of labels & nested labels"
                    assert current_source, "💥 Source for new sentence was not set"
                    sentence = (
                        sentence_counter,
                        {
                            self.config.columns[0]: str(sentence_counter),
                            self.config.columns[1]: current_tokens,
                            self.config.columns[2]: current_labels,
                            self.config.columns[3]: current_nested_labels,
                            self.config.columns[4]: current_source,
                        },
                    )
                    sentence_counter += 1
                    current_tokens = []
                    current_labels = []
                    current_nested_labels = []
                    current_source = ""
                    yield sentence
            # Don't forget last sentence in dataset 🧐
            yield sentence_counter, {
                "id": str(sentence_counter),
                "tokens": current_tokens,
                "labels": current_labels,
                "nested-labels": current_nested_labels,
                "source": current_source,
            }

    @abstractmethod
    def _split_generators(self, dl_manager):
        pass
