# Introduction

Here we add additional debug functionalities to Megatron-LM.
This tool is to provide key fp8 tensor numerical metrics, that can help developer better understand LLM training process.

It has the following features:
* fp8 tensor **quantization errors**: Cosine similarity, Mean Square Error (MSE), Underflows and Overflows
* fp8 tensor **statistics**: abs min and max values
* fp8 **scaling** factors: scale from Delayed Scaling, scale from current tensor abs max.
* Save fp8 GEMM input and output tensors, gradient and model weight into files.
* Selectively choose which TE(transformer engine) layers to log metrics and save tensor.
* Compatible with any version of TE, NeMo, Megatron. (No need to modify TE/NeMo/Megatron source code)

Related implementation can be found in dir `megatron/debug/`
* `hooks.py`: hooks can be added to model to enable debug features.
* `arguments.py`: command line options
* `utils.py`: utilities for debugging

# Usage

1. Copy `megatron/debug` into your megatron folder

2. Use `pretrain_gpt.py` or add API calls in your program, for example:
```python
from megatron.debug.hooks import register_hooks
from megatron.debug.arguments import add_debug_args


def model_provider():
    model = GPTModel(...)
    
    # DEBUG: add log tensor and save tensor hooks
    print_rank_0(f"Set up Log Tensor Hook:\n"
        f"  name pattern: {args.log_tensor_name_pattern}\n"
        f"  interval: {args.log_tensor_interval}")
    
    rank = torch.distributed.get_rank()
    log_fn = lambda string : print(f"[Rank {rank}] {string}")
    matched_modules = register_hooks(model, args,
            args.log_tensor_name_pattern,
            args.log_tensor_interval,
            args.save_tensor,
            args.save_tensor_dir,
            rank,
            log_fn)

    if len(matched_modules) > 0:
        print_rank_0(f"For log tensor name pattern: {args.log_tensor_name_pattern}, find the following layers:")
        for l in matched_modules:
            print_rank_0(f"  {l}")
    else:
        print_rank_0(f"No layers found for the log tensor name pattern: {args.log_tensor_name_pattern}")

if __name__ == "__main__":
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        # DEBUG: add arguments
        extra_args_provider=add_debug_args,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )
```

3. Add following command line options when executing your training, e.g.
```
python pretrain_gpt.py \
    --log-tensor-name-pattern "layers\.(0|31)\D.*(linear_proj).*" \
    --log-tensor-interval 8 \
    --save-tensor \
    --save-tensor-dir exp/saved_tensors
```
This example means it will log and save fp8 GEMM tensors for the layers whose name matching the pattern of `layers\.(0|31)\D.*(linear_proj).*`.
