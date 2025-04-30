import warnings
from typing import Optional, Tuple
import torch

from megatron.core.transformer.moe.triton_permutation import (
    make_row_id_map,
    permute_with_mask_map,
    unpermute_with_mask_map,
)

# import logging
# logger = logging.getLogger(__name__)

# JQ: import float8 tensor
from hybrid import (
    to_float8_tensor,
    is_float8_tensor,
    get_data_from_float8,
    get_scale_from_float8,
)

class _moe_permute_mask_map(torch.autograd.Function):
    """functional Permute with mask router map"""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        routing_map: torch.Tensor,  # [tokens, local_experts]
        num_out_tokens: int,
        probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # JQ: handle fp8 tensor
        assert is_float8_tensor(inp), f"[FP8 ALL2ALL] permute inp should be float8 tensor type!"
        qx = get_data_from_float8(inp)
        sx = get_scale_from_float8(inp)
        scale_hidden_dim = sx.shape[1]

        num_tokens, hidden_size = inp.shape
        num_experts = routing_map.size(1)  # local_experts
        assert num_tokens == sx.shape[0], "scale and input shape mismatch"
        assert inp.size(0) == routing_map.size(0), "Permute not possible"
        assert (
            num_out_tokens is not None
        ), "num_out_tokens must be provided to the fused permute function."

        # JQ: row id map shape [num_experts, tokens]
        row_id_map = make_row_id_map(routing_map, num_tokens, num_experts)

        output, permuted_scale, permuted_probs = permute_with_mask_map(
            qx,
            row_id_map,
            probs,
            sx,
            num_tokens,
            num_experts,
            num_out_tokens,
            hidden_size,
            scale_hidden_dim,
        )

        ctx.save_for_backward(row_id_map)
        ctx.num_experts = num_experts
        ctx.num_tokens = num_tokens
        ctx.hidden_size = hidden_size

        # JQ: return fp8 tensor
        output = to_float8_tensor(output, permuted_scale)

        return output, row_id_map, permuted_probs

    @staticmethod
    def backward(
        ctx,
        permuted_act_grad: torch.Tensor,
        _,
        permuted_probs_grad: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        # pylint: disable=missing-function-docstring
        if not permuted_act_grad.numel():
            return permuted_act_grad, None, None, ctx.probs

        act_grad = None
        probs_grad = None
        if ctx.needs_input_grad[0]:
            (row_id_map,) = ctx.saved_tensors
            assert not isinstance(
                permuted_act_grad, QuantizedTensor
            ), "The backward of moe_permute does not support FP8."
            act_grad, probs_grad = unpermute_with_mask_map(
                permuted_act_grad,
                row_id_map,
                None,
                permuted_probs_grad,
                ctx.num_tokens,
                ctx.num_experts,
                ctx.hidden_size,
            )
        if not ctx.needs_input_grad[3]:
            probs_grad = None
        return act_grad, None, None, probs_grad


class _moe_unpermute_mask_map(torch.autograd.Function):
    """functional Unpermute with mask router map"""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        row_id_map: torch.Tensor,
        merging_probs: Optional[torch.Tensor],
        restore_shape: Optional[torch.Size],
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        if not inp.numel():
            ctx.merging_probs = merging_probs
            return inp

        if restore_shape is None:
            restore_shape = inp.shape
        num_tokens, hidden_size = restore_shape
        num_experts = row_id_map.size(0)

        with_probs = merging_probs is not None
        if with_probs:
            assert merging_probs.is_cuda, "TransformerEngine needs CUDA."

        # Device check
        assert inp.is_cuda, "TransformerEngine needs CUDA."
        assert row_id_map.is_cuda, "TransformerEngine needs CUDA."

        assert not isinstance(
            inp, QuantizedTensor
        ), "The forward of moe_unpermute does not support FP8."
        unpermuted_output, _ = triton_permutation.unpermute_with_mask_map(
            inp,
            row_id_map,
            merging_probs,
            None,
            num_tokens,
            num_experts,
            hidden_size,
        )

        if with_probs:
            ctx.save_for_backward(inp, row_id_map, merging_probs)
        else:
            ctx.save_for_backward(row_id_map)
        ctx.num_experts = num_experts
        ctx.num_tokens = num_tokens
        ctx.num_permuted_tokens = inp.size(0)
        ctx.hidden_size = hidden_size
        ctx.with_probs = with_probs
        return unpermuted_output

    @staticmethod
    def backward(ctx, unpermuted_act_grad):
        # pylint: disable=missing-function-docstring
        if not unpermuted_act_grad.numel():
            return unpermuted_act_grad, None, ctx.merging_probs, None

        act_grad = None
        probs_grad = None
        if ctx.needs_input_grad[0]:
            if ctx.with_probs:
                fwd_input, row_id_map, merging_probs = ctx.saved_tensors
            else:
                (row_id_map,) = ctx.saved_tensors

            fp8 = isinstance(unpermuted_act_grad, QuantizedTensor)
            per_tensor_recipe = isinstance(unpermuted_act_grad, Float8Tensor)
            blockwise_recipe = isinstance(unpermuted_act_grad, Float8BlockwiseQTensor)
            mxfp8_recipe = isinstance(unpermuted_act_grad, MXFP8Tensor)

            if fp8:
                fp8_dtype = unpermuted_act_grad._fp8_dtype
                fake_dtype = unpermuted_act_grad.dtype
                # per-tensor scaling
                if per_tensor_recipe:
                    # Kernel does not need scale in per-tensor scaling
                    fp8_scale = None
                    scale_hidden_dim = None
                    fp8_scale_inv = unpermuted_act_grad._scale_inv
                    unpermuted_act_grad = unpermuted_act_grad._data
                # blockwise scaling
                elif blockwise_recipe:
                    fp8_scale = unpermuted_act_grad._rowwise_scale_inv.T.contiguous()
                    unpermuted_act_grad = unpermuted_act_grad._rowwise_data
                    scale_hidden_dim = fp8_scale.shape[1]
                    assert ctx.num_tokens == fp8_scale.shape[0], "scale and input shape mismatch"
                # mxfp8 scaling
                elif mxfp8_recipe:
                    fp8_scale = unpermuted_act_grad._rowwise_scale_inv.contiguous()
                    unpermuted_act_grad = unpermuted_act_grad._rowwise_data
                    scale_hidden_dim = fp8_scale.shape[1]
                    assert ctx.num_tokens == fp8_scale.shape[0], "scale and input shape mismatch"
                else:
                    raise ValueError("Unsupported FP8 recipe")
            else:
                scale_hidden_dim = None
                fp8_dtype = None
                fp8_scale = None

            if ctx.with_probs:
                assert (
                    not fp8
                ), "The backward of moe_unpermute with merging probs does not support FP8."
                act_grad, probs_grad = (
                    triton_permutation.unpermute_with_mask_map_bwd_with_merging_probs(
                        unpermuted_act_grad,
                        row_id_map,
                        fwd_input,
                        merging_probs,
                        ctx.num_tokens,
                        ctx.num_experts,
                        ctx.num_permuted_tokens,
                        ctx.hidden_size,
                    )
                )
            else:
                act_grad, permuted_scale, _ = triton_permutation.permute_with_mask_map(
                    unpermuted_act_grad,
                    row_id_map,
                    None,
                    fp8_scale,
                    ctx.num_tokens,
                    ctx.num_experts,
                    ctx.num_permuted_tokens,
                    ctx.hidden_size,
                    scale_hidden_dim,
                )

            if fp8:
                if per_tensor_recipe:
                    act_grad = Float8Tensor(
                        data=act_grad,
                        fp8_dtype=fp8_dtype,
                        fp8_scale_inv=fp8_scale_inv,
                        shape=act_grad.shape,
                        dtype=fake_dtype,
                    )
                elif blockwise_recipe:
                    act_grad = Float8BlockwiseQTensor(
                        shape=act_grad.shape,
                        dtype=fake_dtype,
                        rowwise_data=act_grad,
                        rowwise_scale_inv=permuted_scale.T.contiguous(),
                        columnwise_data=None,
                        columnwise_scale_inv=None,
                        fp8_dtype=fp8_dtype,
                        quantizer=None,
                        is_2D_scaled=False,
                        requires_grad=act_grad.requires_grad,
                    )
                elif mxfp8_recipe:
                    act_grad = MXFP8Tensor(
                        shape=act_grad.shape,
                        dtype=fake_dtype,
                        fp8_dtype=fp8_dtype,
                        rowwise_data=act_grad,
                        rowwise_scale_inv=permuted_scale.contiguous(),
                        columnwise_data=None,
                        columnwise_scale_inv=None,
                        quantizer=None,
                        requires_grad=act_grad.requires_grad,
                    )

        if not ctx.needs_input_grad[2]:
            probs_grad = None
        return act_grad, None, probs_grad, None
