#!/bin/bash
python -m pip install transformers==5.0.0
python onnx_exporter_small_all_dynamic.py
#python pt_exporter_small.py
trtexec --onnx=model_dynamic.onnx --explicitBatch --saveEngine=model_bs16_dynamic.plan --minShapes=token_ids:1x64,attn_mask:1x64 --optShapes=token_ids:8x128,attn_mask:8x128 --maxShapes=token_ids:16x256,attn_mask:16x256 --fp16 --verbose --workspace=14000 | tee conversion_bs16_dy.txt
