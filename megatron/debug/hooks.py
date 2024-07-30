import os
import re

import torch
import transformer_engine_extensions as tex
from transformer_engine.common.recipe import Format
from transformer_engine.pytorch.numerics_debug import fp8_tensor_statistics
from transformer_engine.pytorch.module.base import TransformerEngineBaseModule

from megatron.debug.utils import remove_zero_rows, qdq, cosine


def save_tensor_hook(module_name, trainer, save_dir, rank, interval, log_fn, is_fwd=True):
    """Set up hook for forward or backpropagation"""
    if is_fwd:
      tensor_names = ['fwd_x', 'fwd_y']
      fp8_meta_key = "scaling_fwd"
      fp8_gemm_type = tex.FP8FwdTensors.GEMM1_INPUT
    else:
      tensor_names = ['bwd_dx', 'bwd_dy']
      fp8_meta_key = "scaling_bwd"
      fp8_gemm_type = tex.FP8BwdTensors.GRAD_OUTPUT1

    def hook(module, inputs, outputs):
      """Save input and output tensor to file"""
#     step = trainer.global_step + 1   # Use step starting from 1
      step = trainer.curr_iteration + 1   # Use step starting from 1
      if ((step) % interval) == 0:
        tensors = {
          tensor_names[0]: inputs[0].detach().cpu(),   # x or dx
          tensor_names[1]: outputs[0].detach().cpu(),  # y or dy
        }
        # save weight
        if is_fwd:
            tensors['fwd_w'] = list(module.parameters())[0].detach().cpu()  

        # save tensor and scale_inv (if fp8) to file
        for tensor_name, tensor in tensors.items():
          filename = os.path.join(
            save_dir, f'{module_name}.{tensor_name}.step{step}.rank{rank:03d}.pt')
          if not os.path.exists(filename):
            # TODO: only save the first micro batch, for now
            save_obj = { "tensor": tensor }
            if module.fp8 and (tensor_name in ('fwd_x', 'bwd_dy')):
              save_obj["scale"] = module.fp8_meta[fp8_meta_key].scale[fp8_gemm_type]
            torch.save(save_obj, filename)

#           log_fn(f'[save tensor hook] name: {tensor_name}, shape: {tensor.shape}')
            log_fn(f'[save tensor hook] step: {step}, '
              f'layer: {module_name}, save {tensors.keys()} to dir: {save_dir}')

    return hook


def log_tensor_hook(module_name, trainer, interval, log_fn, is_fwd=True):
    """Set up hook for forward or backpropagation"""
    if is_fwd:
      tensor_name = ['fwd_x', 'fwd_w']
      fp8_meta_key = "scaling_fwd"
      fp8_gemm_type = [tex.FP8FwdTensors.GEMM1_INPUT, tex.FP8FwdTensors.GEMM1_WEIGHT]
      fp8_fmt = 'e4m3'
    else:
      tensor_name = ['bwd_dy', ]
      fp8_meta_key = "scaling_bwd"
      fp8_gemm_type = [tex.FP8BwdTensors.GRAD_OUTPUT1, ]
      fp8_fmt = 'e5m2'

    def hook(module, inputs, outputs):
     #step = trainer.global_step + 1   # Use step starting from 1
      step = trainer.curr_iteration + 1   # Use step starting from 1
      if (step % interval) == 0:
	
        # get target tensor: (fwd_x & fwd_w) or bwd_dy
        targets = [inputs[0] if is_fwd else outputs[0], ]
        if is_fwd:
          for p_name, p_tensor in module.named_parameters(recurse=False):
            if p_name == "weight":
              targets.append(p_tensor)

        # process each target tensor
        for index, tensor in enumerate(targets):
          # Remove rows of all zeros (in SFT, we see lots of rows of zero in bwd_dy)
          # tensor = remove_zero_rows(tensor.detach())

          # nonzero abs min and max
          # nonzero_values = tensor[tensor.nonzero(as_tuple=True)]
          # log_fn(f"Nonzero shape: {nonzero_values.shape}")
          nonzero_values = tensor

          amin, amax = torch.abs(nonzero_values).aminmax()
          # scale from current amax
          act_scale = (Format.HYBRID.value.max_fwd if is_fwd else Format.HYBRID.value.max_bwd) / amax

          log_str = (f"{module_name}.{tensor_name[index]}.step{step:d}"
            f", amin: {amin:.3e}"
            f", amax: {amax:.3e}"
            f", amax scale: {act_scale:.3e}")

          # FP8 quantization error
          if module.fp8:
            # Q then DQ
            t_fp8 = qdq(tensor, module.fp8_meta[fp8_meta_key], fp8_gemm_type[index], fp8_fmt)

            # percentage of underflow and overflows
            pct_underflows, pct_overflows = [ 
                value * 100.0 / torch.numel(t_fp8)
                for value in fp8_tensor_statistics(t_fp8, fp8_fmt) ]

            # cos and mse from fp8 quantization
            cos = cosine(tensor, t_fp8)
            mse = torch.nn.functional.mse_loss(tensor, t_fp8)

            # scale from meta
            scale = module.fp8_meta[fp8_meta_key].scale[fp8_gemm_type[index]]

            log_str += (f", meta scale: {scale:.3e}"
                       f", cos: {cos:.3f}"
                       f", mse: {mse:.3e}"
                       f", underflows(%): {pct_underflows:.1f}"
                       f", overflows(%): {pct_overflows:.1f}")

          log_fn(log_str)

    return hook


def register_hooks(model, args, name_pattern, interval, save_tensor,
        save_tensor_dir, rank, log_fn):
    """Register log tensor hook and save tensor hook"""

    matched_modules = []
    for name, layer in model.named_modules():
        if re.search(name_pattern, name) \
            and isinstance(layer, TransformerEngineBaseModule):

            # log tensor hook
            layer.register_forward_hook(log_tensor_hook(name, args,
                interval, log_fn, is_fwd=True))
            layer.register_full_backward_hook(log_tensor_hook(name,
                args, interval, log_fn, is_fwd=False))

            # save tensor hook
            if save_tensor:
                layer.register_forward_hook(save_tensor_hook(
                    name, args, save_tensor_dir, rank,
                    interval, log_fn, is_fwd=True))
                layer.register_full_backward_hook(save_tensor_hook(
                    name, args, save_tensor_dir, rank,
                    interval, log_fn, is_fwd=False))

            matched_modules.append(name)

    return matched_modules
