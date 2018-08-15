# Transformer model for NMT (Tensorflow)
This is implementation of [transformer model](https://arxiv.org/abs/1706.03762), forked from [tensorflow official model](https://github.com/tensorflow/models/tree/master/official/transformer) repsitory.

The official repo does not support pre-generated vocabulary (e.g. bpe or char vocabulary generated by [wmt16_en_de.sh](https://github.com/google/seq2seq/blob/master/bin/data/wmt16_en_de.sh)), as it uses a built-in vocabulary generated by [subtokenizer](https://github.com/tensorflow/models/blob/master/official/transformer/utils/tokenizer.py). This repository supports all these cases.

* Note: I removed some TPU related configuration codes for simplicity.

## Requirements
Please follow the below steps before running models in this repo:

1. Add the top-level ***/nmt-transformer*** folder to the Python path with the command:
   ```
   export PYTHONPATH="$PYTHONPATH:/path/to/nmt-transformer"
   ```
2. Install dependencies:
   ```
   pip3 install --user -r install/requirements.txt
   ```
   or
   ```
   pip install --user -r install/requirements.txt
   ```
   for virtualenv users:
   ```
   pip install -r install/requirements.txt
   ```
   
## Use cases
As I mentioned above, this repository support two use cases.
1. If you want to integrate your own vocabulary file, like bpe vocab, use following scripts.
- [make_tfrecords_subword.py](https://github.com/hkhpub/nmt-transformer/blob/master/make_tfrecords_subword.py)
- [transformer_subword.py](https://github.com/hkhpub/nmt-transformer/blob/master/transformer_subword.py)
- [translate_subword.py](https://github.com/hkhpub/nmt-transformer/blob/master/translate_subword.py)

2. If you want to use built in vocabulary (subtokenizer, same as official codes), use following scripts.
- [make_tfrecords.py](https://github.com/hkhpub/nmt-transformer/blob/master/make_tfrecords.py)
- [transformer_main.py](https://github.com/hkhpub/nmt-transformer/blob/master/transformer_main.py)
- [translate.py](https://github.com/hkhpub/nmt-transformer/blob/master/translate.py)


## Run Configuration
* Note: Only support shared vocabulary file.

1. Export env variables according to your own path.
```sh
PARAM_SET=base
OPUS_DIR=/home/hkh/data/iwslt18/open-subtitles
DATA_DIR=$OPUS_DIR/tf_data_dir
MODEL_DIR=$OPUS_DIR/tf_models/model_eu2en_$PARAM_SET
VOCAB_FILE=$DATA_DIR/vocab.bpe.16000
TEST_SOURCE=$OPUS_DIR/data.tok/test.tok.clean.bpe.16000.eu
TEST_REF=$OPUS_DIR/data.tok/test.tok.clean.en
```

2. Run make_tfrecords_subword.py to convert train, dev data as TFRecord data format.
```sh
python make_tfrecords_subword.py \
--data_dir=$DATA_DIR \
--vocab_prefix=$OPUS_DIR/data.tok/vocab.bpe.16000 \
--train_prefix=$OPUS_DIR/data.tok/train.tok.clean.bpe.16000 \
--dev_prefix=$OPUS_DIR/data.tok/dev.tok.clean.bpe.16000 \
--src=eu \
--tgt=en 
```

3. Run transformer model to train.
```sh

python transformer_subword.py \
--data_dir=$DATA_DIR \
--model_dir=$MODEL_DIR \
--vocab_file=$VOCAB_FILE \
--param_set=$PARAM_SET \
--bleu_source=$TEST_SOURCE \
--bleu_ref=$TEST_REF
```

* For more details, see https://github.com/tensorflow/models/tree/master/official/transformer
