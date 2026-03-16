#!/bin/bash
set -e

pip install transformers==5.0.0

python onnx_exporter_tokenizer.py

trtexec \
  --onnx=model_tokenizer.onnx \
  --saveEngine=model_tokenizer_bs16.plan \
  --minShapes=token_ids:1x64,attn_mask:1x64 \
  --optShapes=token_ids:8x128,attn_mask:8x128 \
  --maxShapes=token_ids:16x256,attn_mask:16x256 \
  --fp16 \
  --verbose \
  --workspace=14000 \
  | tee conversion_tokenizer_bs16.txt

# bert model
mkdir -p /models/bert/1
cp model_tokenizer_bs16.plan /models/bert/1/model.plan

# bert_tokenizer Python backend
mkdir -p /models/bert_tokenizer/1

# bert_ensemble placeholder
mkdir -p /models/bert_ensemble/1
