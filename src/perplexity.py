import os
import argparse

import matplotlib.pyplot as plt
import torch
from tqdm import trange
from transformers import LlamaTokenizer

from homer.embedding import patch_llama_for_yarn
from homer.modeling_llama import LlamaForCausalLM


def gather_logits(logits, input_ids):
    input_ids = input_ids[0, 1 : logits.size(1) + 1].unsqueeze(-1)
    return logits.log_softmax(dim=-1)[0].gather(1, input_ids).squeeze()


def get_accumulated_perplexity(logits):
    accumulated_logits = logits.cumsum(dim=0)
    size = torch.ones_like(accumulated_logits).int().cumsum(dim=0)
    return (-accumulated_logits / size).exp().tolist()


@torch.inference_mode()
def main(args):
    # Setup HOMER arguments
    max_position_id = args.max_position_embeddings * args.scale

    if args.model_type in ["homer", "homer_yarn"]:
        homer_args = {
            "max_chunk_len": max_position_id // 2,
            "max_initial_chunk_len": args.max_initial_chunk_len,
            "reduction_mode": "power_max_last_calibrated",
            "layers_warmup": args.layer_warmup,
            "target_len": max_position_id // 2,
            "bias_path": args.bias_path,
        }
    else:
        homer_args = None

    # Load model
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        homer_args=homer_args,
    )

    # Patch model with YaRN
    if args.model_type in ["yarn", "homer_yarn"]:
        patch_llama_for_yarn(model, args.scale, args.max_position_embeddings, 1024 * 33)

    # Load data
    context = open(args.data_path, "r").read()
    input_ids = tokenizer(context, return_tensors="pt")["input_ids"].to(model.device)

    # Run evaluation
    if args.model_type in ["plain", "yarn"]:
        # Plain models do not require special handling
        input_ids = input_ids[:, : args.max_context_len]
        full_logits = model(input_ids=input_ids).logits.float()
        logits = gather_logits(full_logits, input_ids)

    else:
        # HOMER applies partial evaluation
        evaluation_length = max_position_id // (
            4 if args.model_type == "homer_yarn" else 2
        )

        logits = []

        # First part does not require special handling
        first_part_length = evaluation_length * 2
        first_input_ids = input_ids[:, :first_part_length]

        first_logits = model(input_ids=first_input_ids).logits.float()

        logits.append(gather_logits(first_logits, input_ids))
        del first_logits

        # Following parts
        num_parts = int((args.max_context_len - first_part_length) / evaluation_length)

        offset = first_part_length
        for i in trange(num_parts):
            torch.cuda.empty_cache()
            idx_context_end = offset + i * evaluation_length

            assert (
                input_ids.size(1) > idx_context_end + evaluation_length
            ), "Context too short"

            # Prepare inputs
            prefix_ids = input_ids[:, : args.buffer_len]
            context_ids = input_ids[
                :, args.buffer_len : idx_context_end - args.buffer_len
            ]
            suffix_ids = input_ids[
                :, idx_context_end - args.buffer_len : idx_context_end
            ]

            with torch.no_grad():
                homer_prefix = model.create_homer_prefix(
                    prefix_ids, context_ids, suffix_ids
                )

            # Get logits
            current_input_ids = input_ids[
                :, idx_context_end : idx_context_end + evaluation_length
            ]

            part_logits = model(
                input_ids=current_input_ids,
                homer_prefix=homer_prefix,
            ).logits.float()

            logits.append(
                gather_logits(
                    part_logits[:, -evaluation_length:, :],
                    input_ids[
                        :,
                        idx_context_end:,
                    ],
                )
            )
            del part_logits

        logits = torch.cat(logits, dim=0)

    # Get accumulated perplexity
    acc_perp = get_accumulated_perplexity(logits)

    # Draw and save plots
    plt.plot(acc_perp)
    plt.ylim(3, 12)
    plt.xlabel("Input length")
    plt.ylabel("Perplexity")

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/perplexity.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument(
        "--model_type",
        type=str,
        default="plain",
        choices=["plain", "yarn", "homer", "homer_yarn"],
    )
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--max_position_embeddings", type=int, default=4096)
    parser.add_argument("--gen_length", type=int, default=20)
    # HOMER arguments
    parser.add_argument("--max_initial_chunk_len", type=int, default=-1)
    parser.add_argument("--layer_warmup", type=int, default=12)
    parser.add_argument("--bias_path", type=str, default=None)
    # Task-specific arguments
    parser.add_argument("--buffer_len", type=int, default=100)
    parser.add_argument("--max_context_len", type=int, default=16384)
    args = parser.parse_args()

    main(args)
