import torch
from transformers import BertModel
import argparse
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default="model.pt")
    args = parser.parse_args()

    # Load the BERT model with TorchScript support enabled
    model = BertModel.from_pretrained("bert-base-uncased", torchscript=True)

    # Modify the model to return only the [CLS] token embedding
    class BertClsHead(torch.nn.Module):
        def __init__(self, bert_model):
            super(BertClsHead, self).__init__()
            self.bert = bert_model

        def forward(self, input_ids, attention_mask):
            # Get the encoder outputs from the BERT model
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            # Extract the output for the [CLS] token (first token is at index 0)
            cls_token_output = outputs.last_hidden_state[:, 0, :]
            return cls_token_output

    # Replace the original model with the modified version
    model = BertClsHead(model)
    model = model.eval()
    model.to(device)

    # Create dummy inputs
    bs = 128
    seq_len = 128
    dummy_inputs = (
        torch.randint(1000, (bs, seq_len)).to(device),
        torch.ones(bs, seq_len, dtype=torch.int).to(device),  # Attention mask
    )

    # Trace the modified model with TorchScript
    traced_model = torch.jit.trace(model, dummy_inputs)
    torch.jit.save(traced_model, args.save)
    print("Saved {}".format(args.save))
