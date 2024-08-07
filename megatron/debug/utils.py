import os
import re
import torch

import transformer_engine_extensions as tex
import transformer_engine.pytorch.cpp_extensions as texcpp
from transformer_engine.pytorch.constants import TE_DType


def remove_zero_rows(tensor):
  # Remove all zeros in last dim in a tensor
  shape = tensor.shape
  if len(shape) == 3:
    tensor = tensor.reshape(shape[0] * shape[1], shape[2])
  row_sums = tensor.sum(dim=1)
  nonzero_row_indices = torch.nonzero(row_sums).squeeze()
  return tensor[nonzero_row_indices]


def find_gradients(pl_module, name_pattern, log_fn):
    """print all matching gradients at training start"""
    # Find matching gradient
    targets = list()
#   logging.debug(f"Model weights:")
    for p_name, _ in pl_module.named_parameters(recurse=True):
#       logging.debug(p_name)
        if re.search(name_pattern, p_name):
            targets.append(p_name)

    # Print matching gradient
    if len(targets) > 0:
        log_fn(f"For GRAD name pattern: {name_pattern}, find the following gradients:")
        for g in targets:
            log_fn(f"  {g}")
    else:
        log_fn(f"No gradients found for the given name pattern: {name_pattern}")


def save_weight_or_grad(module, name_pattern, save_dir, step, rank, grad_fn=None):
    if grad_fn:
       _get_fn = grad_fn
       tensor_type = ".grad"
    else:
       _get_fn = lambda x : x
       tensor_type = ""   # param name is in pattern *.weight

    for p_name, param in module.named_parameters(recurse=True):
      if re.search(name_pattern, p_name):
        tensor = _get_fn(param)
        if tensor is not None:
          filename = os.path.join(
            save_dir, f'{p_name}{tensor_type}.step{step}.rank{rank:03d}.pt')
          torch.save(tensor.cpu(), filename)
#       else:
#         logging.warning(f'[DEBUG] {p_name}{tensor_type} has NOT been found!')


def qdq(inp, meta, fp8_tensor, fp8_format="e4m3"):
    """Q and DQ tensor"""
    fp8_type = tex.DType.kFloat8E4M3 if fp8_format == "e4m3" else tex.DType.kFloat8E5M2

    ret = texcpp.cast_to_fp8(inp, meta, fp8_tensor, fp8_type)
    ret = texcpp.cast_from_fp8(ret, meta, fp8_tensor, fp8_type, TE_DType[inp.dtype])

    return ret


def cosine(tensor1, tensor2):
    cosine_sim = torch.nn.functional.cosine_similarity(
        tensor1.view(-1), tensor2.view(-1), dim=0).item()
    return cosine_sim
