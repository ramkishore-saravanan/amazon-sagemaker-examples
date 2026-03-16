import triton_python_backend_utils as pb_utils
import numpy as np
from transformers import BertTokenizer


class TritonPythonModel:
    def initialize(self, args):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def execute(self, requests):
        responses = []
        for request in requests:
            text_tensor = pb_utils.get_input_tensor_by_name(request, "text")
            # shape [-1], each element is a bytes object
            texts = [s.decode("utf-8") for s in text_tensor.as_numpy().flatten()]

            encoded = self.tokenizer(
                texts,
                padding="max_length",
                max_length=128,
                truncation=True,
                return_tensors="np",
            )

            token_ids = encoded["input_ids"].astype(np.int32)
            attn_mask = encoded["attention_mask"].astype(np.int32)

            token_ids_tensor = pb_utils.Tensor("token_ids", token_ids)
            attn_mask_tensor = pb_utils.Tensor("attn_mask", attn_mask)

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[token_ids_tensor, attn_mask_tensor]
                )
            )

        return responses

    def finalize(self):
        pass
