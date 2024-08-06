import argparse
import json
import re

import torch
from tqdm import tqdm
from transformers import LlamaTokenizer

from homer.embedding import patch_llama_for_yarn
from homer.modeling_llama import LlamaForCausalLM


def parse_answer(text):
    # Matching the last digit of the model output
    try:
        response_number = int(re.search(r"\d+", text).group())
    except:
        response_number = -1

    return response_number


@torch.inference_mode()
def generate_homer(model, tokenizer, prefix_ids, context_ids, suffix_ids, gen_length):
    # Create HOMER prefix
    homer_prefix = model.create_homer_prefix(prefix_ids, context_ids, suffix_ids)
    effective_prompt_len = homer_prefix["cache"].key_cache[0].size(2)

    # Generate
    outputs = model.generate(
        homer_prefix=homer_prefix,
        max_new_tokens=gen_length,
    )

    prediction = tokenizer.batch_decode(
        outputs[:, effective_prompt_len:], skip_special_tokens=False
    )[0]

    return prediction


@torch.inference_mode()
def generate_plain(model, tokenizer, prefix_ids, context_ids, suffix_ids, gen_length):
    input_ids = torch.cat([prefix_ids, context_ids, suffix_ids], dim=1)
    effective_prompt_len = input_ids.size(1)

    # Generate
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=gen_length,
    )

    prediction = tokenizer.batch_decode(
        outputs[:, effective_prompt_len:], skip_special_tokens=False
    )[0]

    return prediction


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
    with open(args.data_path, "r") as f:
        dataset = [json.loads(sample) for sample in list(f)]

        if args.num_test_samples > 0:
            dataset = dataset[: args.num_test_samples]

    # Run evaluation
    num_correct = 0
    pbar = tqdm(dataset)
    for idx, sample in enumerate(pbar):
        prefix = "[INST] <<SYS>>\n" + sample["prefix"] + "\n<</SYS>>\n"
        context = sample["context"]
        suffix = sample["postfix"] + " [/INST]"

        prefix_ids = tokenizer(prefix, return_tensors="pt")["input_ids"].to(
            model.device
        )
        context_ids = tokenizer(context, return_tensors="pt")["input_ids"].to(
            model.device
        )[:, 1:]
        suffix_ids = tokenizer(suffix, return_tensors="pt")["input_ids"].to(
            model.device
        )[:, 1:]

        # Leave room for generated tokens
        max_context_len = context_ids.size(1) - args.gen_length
        context_ids = context_ids[:, :max_context_len]

        # Generate prediction
        if args.model_type in ["plain", "yarn"]:
            prediction = generate_plain(
                model, tokenizer, prefix_ids, context_ids, suffix_ids, args.gen_length
            )
        else:
            prediction = generate_homer(
                model, tokenizer, prefix_ids, context_ids, suffix_ids, args.gen_length
            )

        if sample["target"] == parse_answer(prediction):
            num_correct += 1

        pbar.set_postfix({"acc": num_correct / (idx + 1)})

    print(f"Accuracy: {num_correct / len(dataset) * 100:.2f}%")


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
    parser.add_argument("--num_test_samples", type=int, default=-1)
    args = parser.parse_args()

    main(args)
