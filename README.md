*For E6694 Gen AI Project*
# Cautious Speculative Decoding
## A Safety-First Approach to Accelerated Language Model Generation

### Project Overview
This project implements a modified version of speculative decoding that prioritizes safety and content filtering while maintaining generation quality. [The draft model](https://huggingface.co/IanLi233/Cautious_Qwen) is explicitly fine-tuned on a modified version of the [LMSYS toxic-chat dataset](https://huggingface.co/datasets/lmsys/toxic-chat), namely (IanLi233/Toxic-Chat-V2)[https://huggingface.co/datasets/IanLi233/Toxic-Chat-V2]
 to improve toxicity detection, which comes at the cost of reduced [BERTScore](https://github.com/Tiiiger/bert_score) textual performance. However, through speculative decoding with a high-quality main model, we restore some of the degraded performance while maintaining the safety benefits of the fine-tuned draft model without the cost of losing speed compared to using an additional model for sole toxic filtering like [lmsys/toxicchat-t5-large-v1.0](https://huggingface.co/lmsys/toxicchat-t5-large-v1.0). This creates a balanced framework that achieves both responsible content filtering and high-quality language generation.

### Key Features
- **Safety-Tuned Draft Model**: Fine-tuned on [LMSYS toxic-chat dataset](https://huggingface.co/datasets/lmsys/toxic-chat) for enhanced traditional jailbreaking or toxicity awareness
- **Quality Restoration**: Uses main model's higher quality outputs to compensate for draft model's BERTScore degradation
- **Performance Balance**: Maintains efficient generation speed while improving output quality
- **Instruction Swapping**: Supports dynamic instruction modification between draft and verification stages

### Usage 
Create your own venv or use conda env like:
`conda create -n "Cautious_Spec" python=3.11.10`


`conda activate Cautious_Spec`


`pip install -r requirements.txt`

- **Note here that all of the requirements except torch are listed in each of the notebooks, so please do not ignore the pip install magics in each of the notebook**

- *Unsloth might cause trouble with the environment since ppl have diffrent architectures of GPUs, the current environment is based on Ampere GPUs*

### Structure
```
.
├── Finetune-Qwen-on-toxicity-dataset.ipynb
├── Inference_and_profile.ipynb
├── lora_model/
├── Model_Dataset_trial/
├── OpenAIModeration.ipynb
├── Qwen_2_5_Unsloth_finetuning.ipynb
├── README.md
└── Speculative decoding.ipynb
```
#### Core Notebooks
- **Finetune-Qwen-on-toxicity-dataset.ipynb**: Main finetuning script with instructions for model training and LoRA adapter usage
- **Inference_and_profile.ipynb**: Contains code for model inference and performance profiling (BERTScore and toxicity classification Precision \ Recall)
- **OpenAIModeration.ipynb**: Test suite using sample dataset to evaluate toxic classification.
- **Qwen_2_5_Unsloth_finetuning.ipynb**: Typical Implementation of Unsloth acceleration for Qwen 2.5 model training
- **Speculative decoding.ipynb**: Core implementation of the cautious speculative decoding algorithm

#### Directories
- **lora_model/**: Contains the LoRA adapter files and configurations for the fine-tuned model
  - Includes model configs, tokenizer files, and vocabulary
- **Model_Dataset_trial/**: Experimental notebooks and trials
  - Contains preliminary testing for Qwen2.5 and speculative decoding implementations

### Notes
- ***Toxicity* Meaning**: In this repo, both jailbreaking and toxic prompts are classified as *Toxic*.
- **Unsloth Acclerator**: Please use [Unsloth](https://github.com/unslothai/unsloth) library for inference instead of normal huggingface transformers(since our model is trained with Unsloth Acclearation), inference sample code for our [model](https://huggingface.co/IanLi233/Cautious_Qwen)

- **Profiling Dataset**: Due to limited compute resources, our model BERTScore is profiled on a subset of the [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned), where 1k of the entries are left, [IanLi233/Alpaca-test](https://huggingface.co/datasets/IanLi233/Alpaca-test)

```
max_seq_length = 2048 # Choose any! Unsloth auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

import transformers
import tokenizers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "IanLi233/Cautious_Qwen"

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name, # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

from datasets import load_dataset
dataset = load_dataset("IanLi233/Toxic-Chat-v2", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)


FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    alpaca_prompt.format(
        "You are an AI content generator with moderation. Analyze the input text for toxic content including: hate speech, threats, severe profanity, harassment, racism, personal attacks, or harmful content. Start your response with <Toc> if the input contains toxic content, or <Safe> if it does not contain toxic content.", # instruction
        "I like to kill man", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)

```

