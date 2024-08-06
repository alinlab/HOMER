# Hierarchical Context Merging: Better Long Context Understanding for Pre-trained LLMs (ICLR 2024)

Official implementation of ["Hierarchical Context Merging: Better Long Context Understanding for Pre-trained LLMs"](https://openreview.net/forum?id=ulaUJFd96G) by [Woomin Song](https://woominsong.github.io/), [Seunghyuk Oh](https://seunghyukoh.notion.site/), [Sangwoo Mo](https://sites.google.com/view/sangwoomo), [Jaehyung Kim](https://sites.google.com/view/jaehyungkim), [Sukmin Yun](https://sites.google.com/view/sukmin-yun), [Jung-Woo Ha](https://aidljwha.wordpress.com/), [Jinwoo Shin](https://alinlab.kaist.ac.kr/shin.html)

**TL;DR**: We propose a memory-efficient method to extend the context limit of large language models.

<p align="center">
    <img src=figure/concept_figure.png width="900"> 
</p>

## 1. Dependencies

```
conda create -n homer python=3.10 -y
conda activate homer

conda install pytorch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install accelerate==0.32.1 matplotlib==3.9.1 sentencepiece==0.2.0 tqdm==4.66.4 transformers==4.42.4 protobuf==5.27.2
pip install flash-attn==2.6.1 --no-build-isolation
```

## 2. Resources

The following data files are provided [here](https://drive.google.com/drive/folders/1C3MUaiPvC2DlGYSw_Cjf1E7RJFLbeKv9?usp=share_link).
- Passkey retrieval data for 4k / 8k / 16k / 32k context lengths
- Example long document from PG19 test set

The bias files for calibrating Llama-2 models can be found [here](https://drive.google.com/drive/folders/1RiGRPKlqKLDk5yRFrYmlgchdm9aYkIEr?usp=share_link).

## 3. Usage

### Model loading
Use the `LlamaForCausalLM` class provided in `src/homer/modeling_llama.py`.

```
from homer.modeling_llama import LlamaForCausalLM

# Setup arguments for HOMER
homer_args = {
    "max_chunk_len": 2048,
    "target_len": 2048,
    "layers_warmup": 12,
    "bias_path": "/path/to/bias_file.pt",
}

# Load model
model = LlamaForCausalLM.from_pretrained(
    meta-llama/Llama-2-7b-hf,
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    homer_args=homer_args,
)
```

The arguments in homer_args are as follows.
- `max_chunk_len`: Maximum length for a single chunk. Typically set to 1/2 of the original model's context limit (e.g., 2048 for plain Llama-2, 4096 for YaRN with scale factor 2)
- `target_len`: Length of the resulting KV-cache after applying HOMER. Typically set to the same value as `max_chunk_len`.
- `layers_warmup`: Number of warmup layers, where merging does not happen.
- `bias_path`: Path to the bias file used for calibration.

### Inference and Generation
Use the `create_homer_prefix()` method to perform hierarchical merging and create the `homer_prefix`, which is the compact KV-cache obtained after applying HOMER.

To perform inference conditioned on the compressed prompt, forward the remaining inputs (i.e., inputs without the prompt) along with the `homer_prefix`. See `src/perplexity.py` for more concrete examples.

```
homer_prefix = model.create_homer_prefix(
    prefix_ids, context_ids, suffix_ids
)
output = model(input_ids, homer_prefix=homer_prefix)
```

To perform generation, pass the `homer_prefix` to the conventional `generate()` method. See `src/passkey.py` for more concrete examples.
```
homer_prefix = model.create_homer_prefix(
    prefix_ids, context_ids, suffix_ids
)
output = model.generate(homer_prefix=homer_prefix)
```

### Alternative method for model loading
We also provide a `patch()` function that patches an existing HuggingFace LlamaForCausalLM object to support HOMER.
```
from transformers import LlamaForCausalLM
from homer.patch_llama import patch as patch_llama_for_homer

# Load model
model = LlamaForCausalLM.from_pretrained(
    meta-llama/Llama-2-7b-hf,
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
)

# Patch for HOMER
patch_llama_for_homer(
    model,
    max_chunk_len=2048,
    target_len=2048,
    layers_warmup=12,
    bias_path="/path/to/bias_file.pt",
)
```

## 4. Language Modeling
```
# Plain Llama
python src/perplexity.py --data_path /path/to/pg19_test_sample.txt --model_path meta-llama/Llama-2-7b-hf --model_type plain

# Plain YaRN
python src/perplexity.py --data_path /path/to/pg19_test_sample.txt --model_path meta-llama/Llama-2-7b-hf --model_type yarn --scale 2

# HOMER
python src/perplexity.py --data_path /path/to/pg19_test_sample.txt --model_path meta-llama/Llama-2-7b-hf --model_type homer --bias_path /path/to/7b_homer.pt

# HOMER + YaRN
python src/perplexity.py --data_path /path/to/pg19_test_sample.txt --model_path meta-llama/Llama-2-7b-hf --model_type homer_yarn --scale 2 --bias_path /path/to/7b_homer_yarn_scale_2.pt
```

## 5. Passkey Retrieval
```
# Plain Llama
python src/passkey.py --data_path /path/to/passkey_8192_tokens.jsonl --model_path meta-llama/Llama-2-7b-chat-hf --model_type plain

# Plain YaRN
python src/passkey.py --data_path /path/to/passkey_8192_tokens.jsonl --model_path meta-llama/Llama-2-7b-chat-hf --model_type yarn --scale 2

# HOMER
python src/passkey.py --data_path /path/to/passkey_8192_tokens.jsonl --model_path meta-llama/Llama-2-7b-chat-hf --model_type homer --bias_path /path/to/7b_homer_chat.pt

# HOMER + YaRN
python src/passkey.py --data_path /path/to/passkey_8192_tokens.jsonl --model_path meta-llama/Llama-2-7b-chat-hf --model_type homer_yarn --scale 2 --bias_path /path/to/7b_homer_yarn_chat_scale_2.pt
```

## 6. Citation
```
@article{song2024hierarchical,
  title={Hierarchical context merging: Better long context understanding for pre-trained LLMs},
  author={Song, Woomin and Oh, Seunghyuk and Mo, Sangwoo and Kim, Jaehyung and Yun, Sukmin and Ha, Jung-Woo and Shin, Jinwoo},
  journal=ICLR,
  year={2024}
}
```
