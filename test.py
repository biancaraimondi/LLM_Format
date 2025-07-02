from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from peft import PeftModel
import torch
import argparse
import os
import fire
from unsloth import FastLanguageModel, is_bfloat16_supported
import vllm
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_model(lora_path):
    max_seq_length = 2048 # Can increase for longer reasoning traces
    lora_rank = 32 # Larger rank = smarter, but slower
    # Load base model
    #lora_path = model_dir
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = lora_path,
        max_seq_length = max_seq_length,
        load_in_4bit = True, # False for LoRA 16bit
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.2, # Reduce if out of memory
    )
    model.eval()
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def merge_and_save_lora(lora_model_path, output_dir):
    model, tokenizer = get_model(lora_model_path)
    model.save_pretrained_merged(output_dir, tokenizer)


import re
from datasets import load_dataset, Dataset
import ast
import concurrent
import os
import multiprocessing

# os.environ["WANDB_PROJECT"] = "prolog-3b"
# os.environ["WANDB_LOG_MODEL"] = "checkpoint"


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
def get_gsm8k_questions(SYSTEM_PROMPT, split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

def parse_kb(func_def, query, answer):
    from lispy.lispy import standard_env, lisp_eval 
    try:
        env = standard_env()
        print("-"*20)
        print(func_def)
        print("-"*20)
        lisp_eval(func_def, env)
        # Use the instance's query method
        result = lisp_eval(query, env)
        print("Infered: {}, Response: {}, Match: {}".format(result, answer, float(result) == float(answer)))
        try:
            if float(result) == float(answer):
                return 2
        except:
            print("Matching error!")
            return -1
        #print(result)
        # Ensure that the comparison makes sense:
        # This assumes you expect a non-empty result when the answer is correct.
        return -1
    except Exception as e:
        print(f"Error encountered: {e}")
        return -1


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
        print("Timeout reached, terminating process.")
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

def remove_lisp_comments_and_whitespace(code):
    # Remove semicolon comments (from ; to end of line)
    code_no_semicolon = re.sub(r";.*$", "", code, flags=re.MULTILINE)
    # Remove #| ... |# block comments (Lisp block comments)
    code_no_block = re.sub(r"#\|[\s\S]*?\|#", "", code_no_semicolon)
    # Remove newline and tab characters, but keep spaces
    cleaned_code = re.sub(r"[\n\t]", "", code_no_block)
    return cleaned_code

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    knowledge_base = extract_xml_knowledge(r)
    knowledge_base = knowledge_base.replace("```lisp", "")
    knowledge_base = knowledge_base.replace("```", "")
    knowledge_base = remove_lisp_comments_and_whitespace(knowledge_base)
    query = extract_xml_query(r)
    query = query.replace("```lisp", "")
    query = query.replace("```", "")
    query = query.replace("?-", "")
    query = query.strip()
    knowledge_base = remove_lisp_comments_and_whitespace(knowledge_base)
    query = remove_lisp_comments_and_whitespace(query)

    return run_with_timeout(knowledge_base, query, answer)

from tqdm import tqdm
from vllm import LLM, SamplingParams
import pandas as pd


def main(model_B, checkpoint, one_shot, length, dataset):
    SYSTEM_PROMPT = ""
    if one_shot == 0:
        SYSTEM_PROMPT = """
        Generate a lisp solution for the asked question.
        Follow these steps to craft your response:
        1. reason about the given instruction
        2. provide a high-quality lisp solution
        3. write a function call to verify the solution.
        Output in the following format:
        <reasoning>
        ...
        </reasoning>
        <code>
        ...
        </code>
        <funcall>
        ...
        </funcall>

        Write the function call just inside <funcall></funcall> not in <code></code>. Implement the logic in lisp.
        """
    elif one_shot == 1:
        SYSTEM_PROMPT = """
        Generate a lisp solution for the asked question.
        Follow these steps to craft your response:
        1. reason about the given instruction
        2. provide a high-quality lisp solution
        3. write a function call to verify the solution.
        Output in the following format:
        <reasoning>
        ...
        </reasoning>
        <code>
        ...
        </code>
        ...
        </code>
        <funcall>
        ...
        </funcall>

        Write the function call just inside <funcall></funcall> not in <code></code>.

        ## Example
        
        Question:
        James decides to make a bathtub full of jello.  For every pound of water, you need 1.5 tablespoons of jello mix.  The bathtub can hold 6 cubic feet of water.  Each cubic foot of water is 7.5 gallons.  A gallon of water weighs 8 pounds.  A tablespoon of jello mix costs $0.50.  How much did he spend to fill his tub?
        
        <reasoning>
        Here you are reasoning
        </reasoning>
        
        <code>
        (defun jello-cost ()
            (let* ((cubic-feet 6)
                (gallons-per-cubic-foot 7.5)
                (pounds-per-gallon 8)
                (tablespoons-per-pound 1.5)
                (cost-per-tablespoon 0.50)
                (gallons (* cubic-feet gallons-per-cubic-foot))
                (pounds (* gallons pounds-per-gallon))
                (tablespoons (* pounds tablespoons-per-pound))
                (cost (* tablespoons cost-per-tablespoon)))
        cost))
        </code>
        
        <funcall>
        (jello-cost)
        </funcall>
        """
    
    model_B = str(model_B)
    checkpoint = str(checkpoint)
    dataset = str(dataset).strip()
    print("Dataset: ", dataset)
    length = length.replace("_", "") if length == "_" else length
    print("\n\nOne-shot: ", one_shot)
    print("Length: ", length)
    if one_shot == 0:
        one_shot = ""
    elif one_shot == 1:
        one_shot = "_one_shot"
    if checkpoint != "":
        model_dir = "Qwen-" + model_B + "B" + one_shot + length + "/checkpoint-" + checkpoint
        merged_model_dir = "merged_models/" + model_dir
        if not os.path.exists(merged_model_dir):
            print(f"Merging {model_dir}...")
            merge_and_save_lora(model_dir, merged_model_dir)
    else:
        model_dir = "Qwen/Qwen2.5-Coder-" + model_B + "B-Instruct"
        merged_model_dir = model_dir

    if "/check" in model_dir:
        results_dir = "results/"+model_dir.split("/check")[0]
    else:
        if one_shot == "":
            results_dir = "results/Base/zero_shot"
        elif one_shot == "_one_shot":
            results_dir = "results/Base/one_shot"
        elif one_shot == "_five_shot":
            results_dir = "results/Base/five_shot"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if "/check" in model_dir:
        results_dir = results_dir + "/" + model_dir.split("/")[-1] + ".csv"
    else:
        results_dir = results_dir + "/" + model_B + ".csv"
    if dataset == "rosetta2":
        results_dir = results_dir.replace(".csv", "_rosetta2.csv")

    if os.path.exists(results_dir):
        print(f"Results for {model_dir} already exists.")
        return
    else:
        print(f"Generating response for {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(merged_model_dir)
    vllm_model = LLM(model=merged_model_dir, gpu_memory_utilization = 0.4)
    
    list_of_reward = []
    list_of_responses = []

    sampling_params = SamplingParams(
        n = 4,
        best_of=4,
        temperature = 0.1,
        top_p = 0.95,
        top_k=50,
        max_tokens = 1024,
    )

    current_sum = 0
    
    ### ----- TESTING ROSETTA CODE DATASET ----- ###
    questions = []
    if dataset == "rosetta2":
        rosetta_dataset = pd.read_csv('data/prolog_tasks_ground_truth.csv')
        for index, entry in rosetta_dataset.iterrows():
            different_match = []
            different_responses = []

            question_content = entry["task_description"] + "\nTest the algorithm on input " + entry["input_1"]
            questions.append(question_content)
            prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question_content}
            ]

            text = tokenizer.apply_chat_template(prompt,
                                                tokenize = False,
                                                add_generation_prompt = True)
            sentences = vllm_model.generate(
                [text],
                sampling_params=sampling_params,
            )

            for sentence in sentences[0].outputs:
                sentence = sentence.text
                different_match.append(correctness_reward_func(None, sentence, entry["ground_truth_1"]))
                different_responses.append(sentence)
            current_sum += sum(different_match) / len(different_match)
            list_of_reward.append(different_match)
            list_of_responses.append(different_responses)
            collection_results = np.array(list_of_reward)
            print("\nCURRENT ACCURACY: ", (collection_results.sum(1) > 0).sum() / len(collection_results))
    else:
        ### ----- TESTING GSM8K DATASET ----- ###
        dataset = get_gsm8k_questions(SYSTEM_PROMPT, "test")
        for idx, entry in tqdm(enumerate(dataset)):
            different_match = []
            different_responses = []
            prompt = [
                {"role" : "system", "content" : SYSTEM_PROMPT},
                {"role" : "user", "content" : entry["question"]},
            ]
            questions.append(entry["question"])

            text = tokenizer.apply_chat_template(prompt, tokenize = False, add_generation_prompt = True)#, return_tensors = "pt",).to("cuda")
            sentences = vllm_model.generate(
                [text],
                sampling_params=sampling_params,
            )

            """ sentences = []
            inputs =  tokenizer.apply_chat_template(prompt, tokenize = True, add_generation_prompt = True, return_tensors = "pt",).to("cuda")
            outputs = model.generate(
                input_ids = inputs, max_new_tokens = 1024, temperature =0.85, min_p = 0.1,  do_sample=True,         # Enable sampling
            )
            sentences.append(tokenizer.batch_decode(outputs)[-1]) """

            for sentence in sentences[0].outputs:
                sentence = sentence.text
                different_match.append(correctness_reward_func(None, sentence, entry["answer"]))
                different_responses.append(sentence)
            current_sum += sum(different_match) / len(different_match)
            list_of_reward.append(different_match)
            list_of_responses.append(different_responses)

            collection_results = np.array(list_of_reward)
            print("\nCURRENT ACCURACY: ", (collection_results.sum(1) > 0).sum() / len(collection_results))

    # from list_of_reward create a df
    df_reward = pd.DataFrame(list_of_reward, columns=["match_1", "match_2", "match_3", "match_4"])
    df_responses = pd.DataFrame(list_of_responses, columns=["gen_code_1", "gen_code_2", "gen_code_3", "gen_code_4"])
    df = pd.concat([df_reward, df_responses], axis=1)
    df["question"] = questions
    df["sum"] = df["match_1"] + df["match_2"] + df["match_3"] + df["match_4"]
    df["mean"] = df["sum"] / 4
    df["match"] = df["sum"] > 0

    # sort columns in this order: question, sum, mean, match_1, match_2, match_3, match_4, gen_code_1, gen_code_2, gen_code_3, gen_code_4
    df = df[["match", "question", "sum", "mean", "match_1", "match_2", "match_3", "match_4", "gen_code_1", "gen_code_2", "gen_code_3", "gen_code_4"]]

    df.to_csv(results_dir, index=False)

if __name__ == "__main__":
    fire.Fire(main)