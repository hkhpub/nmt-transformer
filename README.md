# nmt-transformer
variant of official transformer model

## Requirements
Please follow the below steps before running models in this repo:

1. Add the top-level ***/nmt-transformer*** folder to the Python path with the command:
   ```
   export PYTHONPATH="$PYTHONPATH:/path/to/nmt-transformer"
   ```
2. Install dependencies:
   ```
   pip3 install --user -r official/requirements.txt
   ```
   or
   ```
   pip install --user -r official/requirements.txt
   ```
   for virtualenv users:
   ```
   pip install -r official/requirements.txt
   ```
   
## Run Configuration
* Note: Only support shared vocabulary file.

1. Exports env variables
```sh
PARAM_SET=base
OPUS_DIR=/home/hkh/data/iwslt18/open-subtitles
DATA_DIR=$OPUS_DIR/tf_data_dir
MODEL_DIR=$OPUS_DIR/tf_models/model_eu2en_$PARAM_SET
VOCAB_FILE=$DATA_DIR/vocab.bpe.16000
TEST_SOURCE=$OPUS_DIR/data.tok/test.tok.clean.bpe.16000.eu
TEST_REF=$OPUS_DIR/data.tok/test.tok.clean.en
```

2. Execute make_tfrecords.py to convert train, dev data as TFRecord data format.
```sh
python make_tfrecords.py \
--data_dir=$DATA_DIR \
--vocab_prefix=$OPUS_DIR/data.tok/vocab.bpe.16000 \
--train_prefix=$OPUS_DIR/data.tok/train.tok.clean.bpe.16000 \
--dev_prefix=$OPUS_DIR/data.tok/dev.tok.clean.bpe.16000 \
--src=eu \
--tgt=en 
```

3. Run transformer model to train
```sh

python transformer_main.py \
--data_dir=$DATA_DIR \
--model_dir=$MODEL_DIR \
--vocab_file=$VOCAB_FILE \
--param_set=$PARAM_SET \
--bleu_source=$TEST_SOURCE \
--bleu_ref=$TEST_REF
```

* For more details, see https://github.com/tensorflow/models/tree/master/official/transformer
