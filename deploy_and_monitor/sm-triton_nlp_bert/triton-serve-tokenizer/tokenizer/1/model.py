import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tokenizers"])

import triton_python_backend_utils as pb_utils
from tokenizers import Tokenizer
import numpy as np
import os

class TritonPythonModel:

    def initialize(self, args):
        # __file__ is /models/tokenizer/1/model.py — tokenizer.json is in the same dir
        model_dir = os.path.dirname(os.path.abspath(__file__))
        self.tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.max_len = 128

    def execute(self, requests):
        responses = []

        for request in requests:
            # Get input text
            text_tensor = pb_utils.get_input_tensor_by_name(request, "text")
            # Flatten to 1D and decode bytes to str
            texts = text_tensor.as_numpy().flatten().tolist()
            texts = [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in texts]

            # Tokenize
            input_ids_list = []
            attention_mask_list = []

            for t in texts:
                enc = self.tokenizer.encode(t)

                ids = enc.ids[:self.max_len]
                mask = [1]*len(ids)

                # pad if needed
                if len(ids) < self.max_len:
                    pad_len = self.max_len - len(ids)
                    ids += [0]*pad_len
                    mask += [0]*pad_len

                input_ids_list.append(ids)
                attention_mask_list.append(mask)

            # Convert to NumPy arrays
            input_ids = np.array(input_ids_list, dtype=np.int32)
            attn_mask = np.array(attention_mask_list, dtype=np.int32)

            # Build Triton response
            outputs = [
                pb_utils.Tensor("token_ids", input_ids),
                pb_utils.Tensor("attn_mask", attn_mask)
            ]

            responses.append(pb_utils.InferenceResponse(output_tensors=outputs))

        return responses