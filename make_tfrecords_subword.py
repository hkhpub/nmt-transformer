import os
import six
import random

from absl import flags
from absl import app as absl_app
import tensorflow as tf
from utils import vocab_utils
from comm_utils.flags import core as flags_core

"""Code Reference: Tensorflow official transformer model"""

# Number of files to split train and dev data
_TRAIN_SHARDS = 100
_DEV_SHARDS = 1
_PREFIX = "open_subtitles18"
_TRAIN_TAG = "train"
_DEV_TAG = "dev"


def shuffle_records(fname):
    """Shuffle records in a single file."""
    tf.logging.info("Shuffling records in file %s" % fname)

    # Rename file prior to shuffling
    tmp_fname = fname + ".unshuffled"
    tf.gfile.Rename(fname, tmp_fname)

    reader = tf.python_io.tf_record_iterator(tmp_fname)
    records = []
    for record in reader:
        records.append(record)
        if len(records) % 100000 == 0:
            tf.logging.info("\tRead: %d", len(records))

    random.shuffle(records)

    # Write shuffled records to original file name
    with tf.python_io.TFRecordWriter(fname) as w:
        for count, record in enumerate(records):
            w.write(record)
            if count > 0 and count % 100000 == 0:
                tf.logging.info("\tWriting record: %d" % count)

    tf.gfile.Remove(tmp_fname)


def shard_filename(path, tag, shard_num, total_shards):
    """Create filename for data shard."""
    return os.path.join(
        path, "%s-%s-%.5d-of-%.5d" % (_PREFIX, tag, shard_num, total_shards))


def all_exist(filepaths):
    """Returns true if all files in the list exist."""
    for fname in filepaths:
        if not tf.gfile.Exists(fname):
            return False
    return True


def make_dir(path):
    if not tf.gfile.Exists(path):
        tf.logging.info("Creating directory %s" % path)
        tf.gfile.MakeDirs(path)


def txt_line_iterator(path):
    """Iterate through lines of file."""
    with tf.gfile.Open(path) as f:
        for line in f:
            yield line.strip()


def dict_to_example(dictionary):
    """Converts a dictionary of string->int to a tf.Example."""
    features = {}
    for k, v in six.iteritems(dictionary):
        features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
    return tf.train.Example(features=tf.train.Features(feature=features))


def encode_and_save_files(vocab_helper, data_dir, src_file, tgt_file, tag, total_shards):
    """Save data from files as encoded Examples in TFrecord format.
    """
    # Create a file for each shard.
    filepaths = [shard_filename(data_dir, tag, n + 1, total_shards)
                 for n in range(total_shards)]

    if all_exist(filepaths):
        tf.logging.info("Files with tag %s already exist." % tag)
        return filepaths

    tf.logging.info("Saving files with tag %s." % tag)

    # Write examples to each shard in round robin order.
    tmp_filepaths = [fname + ".incomplete" for fname in filepaths]
    writers = [tf.python_io.TFRecordWriter(fname) for fname in tmp_filepaths]
    counter, shard = 0, 0
    for counter, (input_line, target_line) in enumerate(zip(txt_line_iterator(src_file), txt_line_iterator(tgt_file))):
        if counter > 0 and counter % 100000 == 0:
            tf.logging.info("\tSaving case %d." % counter)
        example = dict_to_example(
            {"inputs": vocab_helper.encode_source(input_line, add_eos=True),
             "targets": vocab_helper.encode_target(target_line, add_eos=True)})
        writers[shard].write(example.SerializeToString())
        shard = (shard + 1) % total_shards
    for writer in writers:
        writer.close()

    for tmp_name, final_name in zip(tmp_filepaths, filepaths):
        tf.gfile.Rename(tmp_name, final_name)

    tf.logging.info("Saved %d Examples", counter + 1)
    return filepaths


def main(unused_argv):
    make_dir(FLAGS.data_dir)

    train_src_file = FLAGS.train_prefix + "." + FLAGS.src
    train_tgt_file = FLAGS.train_prefix + "." + FLAGS.tgt
    dev_src_file = FLAGS.dev_prefix + "." + FLAGS.src
    dev_tgt_file = FLAGS.dev_prefix + "." + FLAGS.tgt

    if FLAGS.share_vocab:
        src_vocab_file = tgt_vocab_file = FLAGS.vocab_prefix
    else:
        src_vocab_file = FLAGS.vocab_prefix + "." + FLAGS.src
        tgt_vocab_file = FLAGS.vocab_prefix + "." + FLAGS.tgt

    # Source vocab
    src_vocab_size, src_vocab_file = vocab_utils.check_vocab(
        src_vocab_file, FLAGS.data_dir)

    # Target vocab
    if FLAGS.share_vocab:
        print("using source vocab for target")
        tgt_vocab_file = src_vocab_file
        tgt_vocab_size = src_vocab_size
    else:
        tgt_vocab_size, tgt_vocab_file = vocab_utils.check_vocab(
            tgt_vocab_file, FLAGS.data_dir, unk=vocab_utils.UNK)

    tf.logging.info("Encoding files and saving data")
    vocab_helper = vocab_utils.VocabHelper(src_vocab_file, tgt_vocab_file, FLAGS.share_vocab)

    train_tfrecord_files = encode_and_save_files(
        vocab_helper, FLAGS.data_dir, train_src_file, train_tgt_file, _TRAIN_TAG, _TRAIN_SHARDS)

    encode_and_save_files(
        vocab_helper, FLAGS.data_dir, dev_src_file, dev_tgt_file, _DEV_TAG, _DEV_SHARDS)

    for fname in train_tfrecord_files:
        shuffle_records(fname)


def define_data_download_flags():
    flags.DEFINE_string(
        name="data_dir", short_name="dd",
        default="/hdd/data/iwslt18/open-subtitles/tf_data_dir",
        help=flags_core.help_wrap(""))

    flags.DEFINE_string(
        name="vocab_prefix", short_name="vp",
        default="/hdd/data/iwslt18/open-subtitles/data.tok/vocab.bpe.16000",
        help=flags_core.help_wrap(""))

    flags.DEFINE_bool(
        name="share_vocab", short_name="sv",
        default=True,
        help=flags_core.help_wrap(""))

    flags.DEFINE_string(
        name="train_prefix", short_name="tp",
        default="/hdd/data/iwslt18/open-subtitles/data.tok/train.tok.clean.bpe.16000",
        help=flags_core.help_wrap(""))

    flags.DEFINE_string(
        name="dev_prefix", short_name="dp",
        default="/hdd/data/iwslt18/open-subtitles/data.tok/dev.tok.clean.bpe.16000",
        help=flags_core.help_wrap(""))

    flags.DEFINE_string(
        name="src", short_name="src",
        default="eu",
        help=flags_core.help_wrap(""))

    flags.DEFINE_string(
        name="tgt", short_name="tgt",
        default="en",
        help=flags_core.help_wrap(""))


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    define_data_download_flags()
    FLAGS = flags.FLAGS
    absl_app.run(main)
