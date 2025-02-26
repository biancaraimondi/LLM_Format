import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


# Skip restarting message in Colab
import sys; modules = list(sys.modules.keys())
for x in modules: sys.modules.pop(x) if "PIL" in x or "google" in x else None


from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

from unsloth import is_bfloat16_supported
import torch
max_seq_length = 2048 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-Coder-3B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.6, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)

import re
from datasets import load_dataset, Dataset
import ast
import concurrent
import os
from pyswip import Prolog
import multiprocessing

os.environ["WANDB_PROJECT"] = "prolog-3b"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

# Load and prep dataset
SYSTEM_PROMPT = """
Generate a prolog solution for the asked question.
Follow these steps to craft your response:
1. reason about the given instruction
2. provide a high-quality prolog solution
3. write a query to verify the solution.
Output in the following format:
<reasoning>
...
</reasoning>
<code>
...
</code>
<query>
...
</query>

Write the query just inside <query></query> not in <code></code>.
"""

def extract_xml_knowledge(text: str) -> str:
    answer = text.split("<code>")[-1]
    answer = answer.split("</code>")[0]
    return answer.strip()

def extract_xml_query(text: str) -> str:
    answer = text.split("<query>")[-1]
    answer = answer.split("</query>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

dataset = get_gsm8k_questions()


def split_prolog_rules(code):
    """
    Splits a string of multiple Prolog facts and rules by matching
    periods that are not part of a floating-point number. It removes
    any surrounding whitespace from each rule.
    """
    # This regex splits on a period that is NOT immediately followed by a digit.
    # \s* optionally matches any whitespace before and after the period.
    rules = re.split(r'\s*\.(?!\d)\s*', code)
    # Filter out any empty strings that may occur.
    return [rule for rule in rules if rule]

def parse_kb(prolog_code, query, answer):
    reward = 0
    try:
        prolog_interpreter = Prolog()
        # Filter out any empty rule
        #rules = [rule.strip() for rule in prolog_code.split(").") if rule.strip()]
        #rules = [rule + ")" for rule in rules]
        rules = split_prolog_rules(prolog_code)
        print("-"*20)
        print(rules)
        print("-"*20)
        for rule in rules:
            prolog_interpreter.assertz(rule)
            reward += 0.125
        # Use the instance's query method
        result = list(prolog_interpreter.query(query))
        for inference in result:
          for _, result_inference in inference.items():
            print("Infered: {}, Response: {}, Match: {}".format(result_inference, answer, float(result_inference) == float(answer)))
            try:
              if float(result_inference) == float(answer):
                return reward + 1
            except:
              print("Matching error!")
              return reward
        #print(result)
        # Ensure that the comparison makes sense:
        # This assumes you expect a non-empty result when the answer is correct.
        return reward + 0.25
    except Exception as e:
        print(f"Error encountered: {e}")
        return reward

def worker(prolog_code, query, answer):
  try:
    return parse_kb(prolog_code, query, answer)
  except:
    return 0

def eval_with_timeout(prolog_code, query, answer, env=None, timeout=5):
    if env is None:
        env = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(worker, prolog_code, query, answer)
        result = future.result(timeout=timeout)
        return 0 if result is None else result


def remove_prolog_comments_and_whitespace(code):
    # Remove block comments (/* ... */)
    code_no_block = re.sub(r'/\*[\s\S]*?\*/', '', code)
    # Remove single-line comments (% ...) from each line
    code_no_comments = re.sub(r'(?m)%.*$', '', code_no_block)
    # Remove newline and tab characters, but keep spaces
    cleaned_code = re.sub(r'[\n\t]', '', code_no_comments)
    return cleaned_code

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    reward = []
    for r, a in zip(responses, answer):
      knowledge_base = extract_xml_knowledge(r)
      knowledge_base = knowledge_base.replace("```prolog", "")
      knowledge_base = knowledge_base.replace("```", "")
      query = extract_xml_query(r)
      query = query.replace("```prolog", "")
      query = query.replace("```", "")
      query = query.replace("?-", "")
      query = query.strip()
      knowledge_base = remove_prolog_comments_and_whitespace(knowledge_base)
      print("#"*20)
      print(knowledge_base)
      print("#"*20)
      query = remove_prolog_comments_and_whitespace(query)
      reward_achieved = eval_with_timeout(knowledge_base, query, a)
      # check with ast if code can be parsed
      reward.append(reward_achieved)
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[-1]}", f"\nResponse:\n{responses[-1]}")
    return reward

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\s*([\s\S]*?)\s*</reasoning>\s*<code>\s*([\s\S]*?)\s*</code>\s*<query>\s*([\s\S]*?)\s*</query>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>\s*([\s\S]*?)\s*</reasoning>\s*<code>\s*([\s\S]*?)\s*</code>\s*<query>\s*([\s\S]*?)\s*</query>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("<code>\n") == 1:
        count += 0.125
    if text.count("\n</code>\n") == 1:
        count += 0.125
    if text.count("\n<query>\n") == 1:
        count += 0.125
    if text.count("\n</query>") == 1:
        count += 0.125
        count -= (len(text.split("\n</query>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4, # Increase to 4 for smoother training
    num_generations = 8, # Decrease if out of memory
    max_prompt_length = 256,
    max_completion_length = 1024,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 1000,
    save_steps = 100,
    max_grad_norm = 0.1,
    report_to = "wandb", # Can use Weights & Biases
    output_dir="training_output",
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        correctness_reward_func,
        strict_format_reward_func,
        soft_format_reward_func
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()

# text = tokenizer.apply_chat_template([
#     {"role" : "user", "content" : "Calculate pi."},
# ], tokenize = False, add_generation_prompt = True)

# from vllm import SamplingParams
# sampling_params = SamplingParams(
#     temperature = 0.8,
#     top_p = 0.95,
#     max_tokens = 1024,
# )
# output = model.fast_generate(
#     [text],
#     sampling_params = sampling_params,
#     lora_request = None,
# )[0].outputs[0].text

# output

# model.save_lora("grpo_saved_lora")

# text = tokenizer.apply_chat_template([
#     {"role" : "system", "content" : SYSTEM_PROMPT},
#     {"role" : "user", "content" : "Calculate pi."},
# ], tokenize = False, add_generation_prompt = True)

# from vllm import SamplingParams
# sampling_params = SamplingParams(
#     temperature = 0.8,
#     top_p = 0.95,
#     max_tokens = 1024,
# )
# output = model.fast_generate(
#     text,
#     sampling_params = sampling_params,
#     lora_request = model.load_lora("grpo_saved_lora"),
# )[0].outputs[0].text

# output

# # Merge to 16bit
# if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
# if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

# # Merge to 4bit
# if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
# if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")

# # Just LoRA adapters
# if False: model.save_pretrained_merged("model", tokenizer, save_method = "lora",)
# if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "lora", token = "")

# # Save to 8bit Q8_0
# if False: model.save_pretrained_gguf("model", tokenizer,)
# # Remember to go to https://huggingface.co/settings/tokens for a token!
# # And change hf to your username!
# if False: model.push_to_hub_gguf("hf/model", tokenizer, token = "")

# # Save to 16bit GGUF
# if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
# if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")

# # Save to q4_k_m GGUF
# if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
# if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")

# # Save to multiple GGUF options - much faster if you want multiple!
# if False:
#     model.push_to_hub_gguf(
#         "hf/model", # Change hf to your username!
#         tokenizer,
#         quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
#         token = "",
#     )
