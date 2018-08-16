# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility to handle vocabularies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import os
import tensorflow as tf

UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
UNK_ID = 0
SOS_ID = 1
EOS_ID = 2


class VocabHelper(object):

    def __init__(self, src_vocab_file, tgt_vocab_file=None, share_vocab=True):
        self.src_vocab_table, self.tgt_vocab_table = create_vocab_tables(
            src_vocab_file, tgt_vocab_file, share_vocab)
        self.reverse_tgt_vocab_table = reverse_vocab_table(self.tgt_vocab_table)

        pass

    def encode(self, raw_string, add_eos=False):
        return _encode(self.src_vocab_table, raw_string, add_eos)

    def encode_source(self, raw_string, add_eos=False):
        return _encode(self.src_vocab_table, raw_string, add_eos)

    def encode_target(self, raw_string, add_eos=False):
        return _encode(self.tgt_vocab_table, raw_string, add_eos)

    def decode(self, ids):
        return " ".join([self.reverse_tgt_vocab_table[token_id]
                         if token_id in self.reverse_tgt_vocab_table else UNK
                         for token_id in ids])


def _encode(src_vocab_table, raw_string, add_eos=False):
    token_ids = [src_vocab_table[token] if token in src_vocab_table else UNK_ID
                 for token in raw_string.split(" ")]
    if add_eos:
        token_ids.append(EOS_ID)
    return token_ids


def load_vocab(vocab_file):
    vocab = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
        vocab_size = 0
        for word in f:
            vocab_size += 1
            vocab.append(word.strip())
    return vocab, vocab_size


def check_vocab(vocab_file, out_dir, check_special_token=True, sos=None,
                eos=None, unk=None):
    """Check if vocab_file doesn't exist, create from corpus_file."""
    if tf.gfile.Exists(vocab_file):
        print("# Vocab file %s exists" % vocab_file)
        vocab, vocab_size = load_vocab(vocab_file)
        if check_special_token:
            # Verify if the vocab starts with unk, sos, eos
            # If not, prepend those tokens & generate a new vocab file
            if not unk: unk = UNK
            if not sos: sos = SOS
            if not eos: eos = EOS
            assert len(vocab) >= 3
            if vocab[0] != unk or vocab[1] != sos or vocab[2] != eos:
                print("The first 3 vocab words [%s, %s, %s]"
                      " are not [%s, %s, %s]" %
                      (vocab[0], vocab[1], vocab[2], unk, sos, eos))
                vocab = [unk, sos, eos] + vocab
                vocab_size += 3
                new_vocab_file = os.path.join(out_dir, os.path.basename(vocab_file))
                with codecs.getwriter("utf-8")(
                    tf.gfile.GFile(new_vocab_file, "wb")) as f:
                    for word in vocab:
                        f.write("%s\n" % word)
                vocab_file = new_vocab_file
    else:
        raise ValueError("vocab_file '%s' does not exist." % vocab_file)

    vocab_size = len(vocab)
    return vocab_size, vocab_file


def reverse_vocab_table(vocab_table):
    return dict(zip(vocab_table.values(), vocab_table.keys()))


def create_vocab_tables(src_vocab_file, tgt_vocab_file, share_vocab):
    """Creates vocab tables for src_vocab_file and tgt_vocab_file."""
    src_vocab_table = dict()
    for token in load_vocab(src_vocab_file)[0]:
        src_vocab_table[token] = len(src_vocab_table)

    if share_vocab:
        tgt_vocab_table = src_vocab_table
    else:
        tgt_vocab_table = dict()
        for token in load_vocab(tgt_vocab_file)[0]:
            tgt_vocab_table[token] = len(tgt_vocab_table)
    return src_vocab_table, tgt_vocab_table
