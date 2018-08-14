# nmt-transformer
modified version of official transformer model

run configuration

```sh
PARAM_SET=hkh
DATA_DIR=/hdd/data/iwslt18/open-subtitles/tf_data_dir
MODEL_DIR=/hdd/data/iwslt18/open-subtitles/tf_models/model_eu2en_$PARAM_SET
VOCAB_FILE=$DATA_DIR/vocab.bpe.16000
TEST_SOURCE=/hdd/data/iwslt18/open-subtitles/data.tok/test.tok.clean.bpe.16000.eu
TEST_REF=/hdd/data/iwslt18/open-subtitles/data.tok/test.tok.clean.en

python transformer_main.py \
--data_dir=$DATA_DIR \
--model_dir=$MODEL_DIR \
--vocab_file=$VOCAB_FILE \
--param_set=$PARAM_SET \
--bleu_source=$TEST_SOURCE \
--bleu_ref=$TEST_REF
```
