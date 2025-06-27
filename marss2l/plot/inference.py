import torch
from typing import Callable, Dict
from numpy.typing import NDArray

KEYS_FROM_ITEM = ["site_ids", "y_context_ls0_0"]


def inference_function_from_torch_model(model:torch.nn.Module, keys_from_item:str=None) -> Callable[[Dict[str, NDArray]], NDArray]:
    if keys_from_item is None:
        keys_from_item = KEYS_FROM_ITEM   
    
    def inference_function(item:Dict[str, torch.Tensor]) -> NDArray:
        model.eval()
        with torch.no_grad():
            batch = {key: item[key].unsqueeze(0) for key in keys_from_item}
            out = model(batch)
            out = torch.sigmoid(out).squeeze(1) # (B, 1, H, W) -> (B, H, W)
            preds = out[0].numpy()

        return preds
    return inference_function
