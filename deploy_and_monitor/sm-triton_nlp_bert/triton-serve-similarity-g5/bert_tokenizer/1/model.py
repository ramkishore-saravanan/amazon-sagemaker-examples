import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tokenizers"])

import triton_python_backend_utils as pb_utils
from tokenizers import Tokenizer
import numpy as np
import os


class TritonPythonModel:
    def initialize(self, args):
        model_dir = os.path.dirname(os.path.abspath(__file__))
        self.tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.max_len = 30

    def execute(self, requests):
        responses = []
        for request in requests:
            text_tensor = pb_utils.get_input_tensor_by_name(request, "text")
            texts = text_tensor.as_numpy().flatten().tolist()
            texts = [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in texts]

            input_ids_list = []
            attention_mask_list = []

            for t in texts:
                enc = self.tokenizer.encode(t)

                ids = enc.ids[:self.max_len]
                mask = [1] * len(ids)

                if len(ids) < self.max_len:
                    pad_len = self.max_len - len(ids)
                    ids += [0] * pad_len
                    mask += [0] * pad_len

                input_ids_list.append(ids)
                attention_mask_list.append(mask)

            token_ids = np.array(input_ids_list, dtype=np.int32)
            attn_mask = np.array(attention_mask_list, dtype=np.int32)

            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("token_ids", token_ids),
                pb_utils.Tensor("attn_mask", attn_mask),
            ]))

        return responses

    def finalize(self):
        pass
