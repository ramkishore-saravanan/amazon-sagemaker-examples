#!/bin/bash
python -m pip install transformers==5.0.0
python onnx_exporter_small.py
#python pt_exporter_small.py
trtexec --onnx=model.onnx --explicitBatch --saveEngine=model_bs16.plan --minShapes=token_ids:1x128,attn_mask:1x128 --optShapes=token_ids:8x128,attn_mask:8x128 --maxShapes=token_ids:16x128,attn_mask:16x128 --fp16 --verbose --workspace=14000 | tee conversion_bs16_dy.txt
