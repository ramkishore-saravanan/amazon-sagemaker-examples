import torch
from transformers import BertModel
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default="model_dynamic.onnx")
    args = parser.parse_args()

    model = BertModel.from_pretrained("bert-base-uncased")
    model.config._attn_implementation = "eager"

    # Modify the model to return only the [CLS] token embedding
    class BertClsHead(torch.nn.Module):
        def __init__(self, bert_model):
            super(BertClsHead, self).__init__()
            self.bert = bert_model.half()

        def forward(self, input_ids, attention_mask):
            # Get the encoder outputs from the BERT model
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            # Extract the output for the [CLS] token (first token is at index 0)
            cls_token_output = outputs[0][:, 0, :].float()
            return cls_token_output

    # Replace the original model with the modified version
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BertClsHead(model)
    model = model.eval()
    model.to(device)

    bs = 2
    seq_len = 128
    dummy_inputs = (
        torch.randint(1000, (1,128), dtype=torch.long).to(device),
        torch.ones(1,128, dtype=torch.long).to(device)
    )
    
    torch.onnx.export(
        model,
        dummy_inputs,
        args.save,
        export_params=True,
        opset_version=17,
        input_names=["token_ids", "attn_mask"],
        output_names=["output"],
      dynamic_axes= {
          "token_ids": {0: "batch_size", 1: "seq_len"},
          "attn_mask": {0: "batch_size", 1: "seq_len"},
          "output": {0: "batch_size"},
      }
    )

    print("Saved {}".format(args.save))
