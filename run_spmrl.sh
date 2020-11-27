#!/usr/bin/env bash
#python main_largest_phrase_spmrl_transformer_tunning_external_embedding.py --external_embedding char_lstm --language $parsing_language
export RUN_LANGUGE='Polish'
python main_largest_phrase_spmrl_transformer_tunning_external_embedding_predpos.py --external_embedding bert \
--use_tags --language $RUN_LANGUGE --bert_model bert-base-multilingual-cased \
--tunning_dict tunning_dict_mix_attention_external_embedding_bert.txt