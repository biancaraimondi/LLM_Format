from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import argparse
import os
import fire

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = "cuda" if torch.cuda.is_available() else "cpu"

def merge_and_save_lora(tokenizer, model, output_dir: str):
    """Merge LoRA weights and save model"""
    print("Starting to merge LoRA weights...")
    # Merge LoRA weights into base model
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to: {output_dir}")
    # Save merged model
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=True
    )
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    print("Model saving completed!")


import re
from datasets import load_dataset, Dataset
import ast
import concurrent
import os
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

Write the query just inside <query></query> not in <code></code>. Implement the logic in prolog.
"""

def get_structured_output(output, idx):
  return [
        {"content": output[0].outputs[0].text}
    ]

def get_structured_question(output):
  return [
    [
        {"content": output[0]}
    ]
  ]

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
    from pyswip import Prolog
    try:
        prolog_interpreter = Prolog()
        # Filter out any empty rule
        #rules = [rule.strip() for rule in prolog_code.split(").") if rule.strip()]
        #rules = [rule + ")" for rule in rules]
        rules = split_prolog_rules(prolog_code)
        # print("-"*20)
        # print(rules)
        # print("-"*20)
        for rule in rules:
            if "?-" in rule:
              continue
            prolog_interpreter.assertz(rule)
        # Use the instance's query method
        result = list(prolog_interpreter.query(query))
        # print("-"*20)
        # print(result)
        # print("-"*20)
        for inference in result:
          for _, result_inference in inference.items():
            #print("Infered: {}, Response: {}, Match: {}".format(result_inference, answer, float(result_inference) == float(answer)))
            try:
              if float(result_inference) == float(answer):
                return 1
            except:
              #print("Matching error!")
              return 0
        #print(result)
        # Ensure that the comparison makes sense:
        # This assumes you expect a non-empty result when the answer is correct.
        return 0
    except Exception as e:
        #print(f"Error encountered: {e}")
        return 0


def worker(q, prolog_code, query, answer):
    result = None
    try:
        # Assuming parse_kb returns the desired result
        result = parse_kb(prolog_code, query, answer)
    except Exception as e:
        result = 0
    # Put the result into the queue for the parent to retrieve
    q.put(result)

def run_with_timeout(prolog_code, query, answer, timeout=5):
    # Create a queue to share data between processes
    q = multiprocessing.Queue()
    proc = multiprocessing.Process(target=worker, args=(q, prolog_code, query, answer))
    proc.start()
    # Wait for the process to finish or timeout
    proc.join(timeout)

    if proc.is_alive():
        #print("Timeout reached, terminating process.")
        proc.terminate()
        proc.join()
        return 0

    result= None

    try:
        # Retrieve the result from the queue
        result = q.get(timeout=1)
    except Exception as e:
        # If the queue is empty or another error occurs, return a default value
        result = 0

    return result


def remove_prolog_comments_and_whitespace(code):
    # Remove block comments (/* ... */)
    code_no_block = re.sub(r'/\*[\s\S]*?\*/', '', code)
    # Remove single-line comments (% ...) from each line
    code_no_comments = re.sub(r'(?m)%.*$', '', code_no_block)
    # Remove newline and tab characters, but keep spaces
    cleaned_code = re.sub(r'[\n\t]', '', code_no_comments)
    return cleaned_code

# Reward functions
def correctness_reward_func(prompts, response, answer, **kwargs) -> list[float]:
    knowledge_base = extract_xml_knowledge(response)
    knowledge_base = knowledge_base.replace("```prolog", "")
    knowledge_base = knowledge_base.replace("```", "")
    query = extract_xml_query(response)
    query = query.replace("```prolog", "")
    query = query.replace("```", "")
    query = query.replace("?-", "")
    query = query.strip()
    knowledge_base = remove_prolog_comments_and_whitespace(knowledge_base)
    query = remove_prolog_comments_and_whitespace(query)
    return run_with_timeout(knowledge_base, query, answer)







from tqdm import tqdm
from vllm import LLM, SamplingParams
import pandas as pd


def main(model_B, checkpoint, one_shot):
    model_B = str(model_B)
    checkpoint = str(checkpoint)
    one_shot = "" if one_shot == "False" else "_one_shot"
    pretrained_model = "Qwen/Qwen2.5-Coder-" + model_B + "B-Instruct"
    model_dir = "Qwen-" + model_B + "B" + one_shot + "/checkpoint-" + checkpoint
    merged_model_dir = "merged_models/" + model_dir

    print(f"Generating response for {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model,
            trust_remote_code=True,
        )

    vllm_model = LLM(model=merged_model_dir)






    dataset = get_gsm8k_questions("test")
    list_of_reward = []
    list_of_responses = []

    sampling_params = SamplingParams(
        n = 4,
        best_of=4,
        temperature = 0.8,
        top_p = 0.95,
        top_k=50,
        max_tokens = 1024,
    )

    for idx, entry in tqdm(enumerate(dataset)):
        different_match = []
        different_responses = []
        prompt = [
            {"role" : "system", "content" : SYSTEM_PROMPT},
            {"role" : "user", "content" : entry["question"]},
        ]

        text = tokenizer.apply_chat_template(prompt, tokenize = False, add_generation_prompt = True)#, return_tensors = "pt",).to("cuda")

        sentences = vllm_model.generate(
            [text],
            sampling_params=sampling_params,
        )

        for sentence in sentences[0].outputs:
            sentence = sentence.text
            different_match.append(correctness_reward_func(text, sentence, entry["answer"]))
            different_responses.append(sentence)
        list_of_reward.append(different_match)
        list_of_responses.append(different_responses)

    # from list_of_reward create a df
    df_reward = pd.DataFrame(list_of_reward, columns=["match_1", "match_2", "match_3", "match_4"])
    df_responses = pd.DataFrame(list_of_responses, columns=["gen_code_1", "gen_code_2", "gen_code_3", "gen_code_4"])
    df = pd.concat([df_reward, df_responses], axis=1)
    df["question"] = dataset["question"]
    df["sum"] = df["match_1"] + df["match_2"] + df["match_3"] + df["match_4"]
    df["mean"] = df["sum"] / 4

    results_dir = "results/"+model_dir.split("/check")[0]
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_dir = results_dir + "/" + model_dir.split("/")[-1] + ".csv"

    # sort columns in this order: question, sum, mean, match_1, match_2, match_3, match_4, gen_code_1, gen_code_2, gen_code_3, gen_code_4
    df = df[["question", "sum", "mean", "match_1", "match_2", "match_3", "match_4", "gen_code_1", "gen_code_2", "gen_code_3", "gen_code_4"]]

    df.to_csv(results_dir, index=False)

if __name__ == "__main__":
    fire.Fire(main)