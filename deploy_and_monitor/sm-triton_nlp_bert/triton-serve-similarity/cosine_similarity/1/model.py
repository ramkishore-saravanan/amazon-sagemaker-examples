import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            embeddings = pb_utils.get_input_tensor_by_name(request, "embeddings").as_numpy()
            # embeddings shape: [N, 768]
            query = embeddings[0]       # [768]
            products = embeddings[1:]   # [N-1, 768]

            # L2-normalise
            query_norm = query / (np.linalg.norm(query) + 1e-9)
            products_norm = products / (np.linalg.norm(products, axis=1, keepdims=True) + 1e-9)

            scores = products_norm @ query_norm  # [N-1]

            out_tensor = pb_utils.Tensor("scores", scores.astype(np.float32))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
        return responses
