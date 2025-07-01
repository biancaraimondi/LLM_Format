import fire
import wandb

# run = wandb.init(
#     # set the wandb project where this run will be logged
#     project="my-awesome-project",
#     # track hyperparameters and run metadata
#     config=param,
# )

def main(model_B, one_shot):
    model_B = str(model_B)
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

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from unsloth import FastLanguageModel, PatchFastRL
    PatchFastRL("GRPO", FastLanguageModel)

    from unsloth import is_bfloat16_supported
    import torch
    max_seq_length = 2048 # Can increase for longer reasoning traces
    lora_rank = 32 # Larger rank = smarter, but slower

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "Qwen/Qwen2.5-Coder-"+model_B+"B-Instruct",
        max_seq_length = max_seq_length,
        load_in_4bit = True, # False for LoRA 16bit
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.2, # Reduce if out of memory
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
    import multiprocessing

    os.environ["WANDB_PROJECT"] = "lisp_length_no_kl"
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    os.environ["WANDB_ENTITY"] = "halykoss"


    def extract_xml_knowledge(text: str) -> str:
        answer = text.split("<code>")[-1]
        answer = answer.split("</code>")[0]
        return answer.strip()

    def extract_xml_query(text: str) -> str:
        answer = text.split("<funcall>")[-1]
        answer = answer.split("</funcall>")[0]
        return answer.strip()
    
    def extract_xml_reasoning(text: str) -> str:
        answer = text.split("<reasoning>")[-1]
        answer = answer.split("</reasoning>")[0]
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
        responses = [completion[0]['content'] for completion in completions]
        q = prompts[0][-1]['content']
        reward = []
        for r, a in zip(responses, answer):
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
            print("#"*20)
            print(knowledge_base)
            print("#"*20)
            query = remove_lisp_comments_and_whitespace(query)
            reward_achieved = run_with_timeout(knowledge_base, query, a)
            # check with ast if code can be parsed
            reward.append(reward_achieved)
        print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[-1]}", f"\nResponse:\n{responses[-1]}")
        return reward
    
    def count_reasoning(completions, **kwargs) -> list[float]:
        """Reward function that checks the length of the reasoning section. It assigns a reward for every row in the reasoning section of 0.001."""
        contents = [completion[0]["content"] for completion in completions]
        lengths = []
        for c in contents:
            reasoning = extract_xml_reasoning(c)
            length = len(reasoning.split())
            lengths.append(length*0.0001)
        return lengths
    
    def count_code(completions, **kwargs) -> list[float]:
        """Reward function that checks the length of the code section. It assigns a reward for every row in the code section of 0.001."""
        contents = [completion[0]["content"] for completion in completions]
        lengths = []
        for c in contents:
            code = extract_xml_knowledge(c)
            length = len(code.split())
            lengths.append(length*0.0001)
        return lengths
    
    def length_correctness_code_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
        """Reward function that ensure code length is less than x while being correct."""
        corr = correctness_reward_func(prompts, completions, answer)
        code = count_code(completions)
        reward = []
        for i in range(len(completions)):
            if code[i] > 0.003 and code[i] < 0.005 and corr[i] == 2:
                reward.append(1)
            else:
                reward.append(0)
        return reward
    
    def length_correctness_reasoning_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
        """Reward function that ensure code length is less than x while being correct."""
        corr = correctness_reward_func(prompts, completions, answer)
        reasoning = count_reasoning(completions)
        reward = []
        for i in range(len(completions)):
            if reasoning[i] > 0.009 and reasoning[i] < 0.013 and corr[i] == 2:
                reward.append(1)
            else:
                reward.append(0)
        return reward

    def strict_format_reward_func(completions, **kwargs) -> list[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r"^<reasoning>\s*([\s\S]*?)\s*</reasoning>\s*<code>\s*([\s\S]*?)\s*</code>\s*<query>\s*([\s\S]*?)\s*</query>\n$"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r) for r in responses]
        return [0.5 if match else 0.0 for match in matches]

    def soft_format_reward_func(completions, **kwargs) -> list[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r"<reasoning>\s*([\s\S]*?)\s*</reasoning>\s*<code>\s*([\s\S]*?)\s*</code>\s*<funcall>\s*([\s\S]*?)\s*</funcall>"
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
            # if "?-" in text between <code> and </code> then reward -0.5
            code_text = text.split("<code>\n")[-1]
            code_text = code_text.split("\n</code>")[0]
            query_text = text.split("<funcall>\n")[-1]
            query_text = query_text.split("\n</funcall>")[0]
            if "?-" in code_text or query_text.replace("\n", "") in code_text:
                print("\nFOUND QUERY IN CODE: ", query_text.replace("\n", ""))
                count -= 0.5
        if text.count("\n</code>\n") == 1:
            count += 0.125
        if text.count("\n<funcall>\n") == 1:
            count += 0.125
            #count -= len(text.split("\n</query>\n")[-1])*0.001
        if text.count("\n</funcall>\n") == 1:
            count += 0.125
            #count -= (len(text.split("\n</query>")[-1]) - 1)*0.001
        return count

    def xmlcount_reward_func(completions, **kwargs) -> list[float]:
        contents = [completion[0]["content"] for completion in completions]
        return [count_xml(c) for c in contents]


    from trl import GRPOConfig, GRPOTrainer
    shot_string = ""
    if one_shot == 1:
        shot_string = "_one_shot"
    elif one_shot == 5:
        shot_string = "_five_shot"
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
        max_prompt_length = 512,
        max_completion_length = 1024,
        beta=0.0,
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps = 1500,
        save_steps = 500,
        max_grad_norm = 0.1,
        report_to = "wandb", # Can use Weights & Biases
        output_dir="Qwen-"+model_B+"B"+shot_string+"_length",
    )


    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            xmlcount_reward_func,
            correctness_reward_func,
            # strict_format_reward_func,
            soft_format_reward_func,
            #count_reasoning,
            #count_code,
            #length_correctness_reasoning_reward_func,
        ],
        args = training_args,
        train_dataset = dataset,
    )
    trainer.train() # resume_from_checkpoint = True


    text = tokenizer.apply_chat_template([
        {"role" : "user", "content" : "Calculate pi."},
    ], tokenize = False, add_generation_prompt = True)

    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature = 0.8,
        top_p = 0.95,
        max_tokens = 1024,
    )
    output = model.fast_generate(
        [text],
        sampling_params = sampling_params,
        lora_request = None,
    )[0].outputs[0].text

    output

    model.save_lora("grpo_saved_lora")

    text = tokenizer.apply_chat_template([
        {"role" : "system", "content" : SYSTEM_PROMPT},
        {"role" : "user", "content" : "Calculate pi."},
    ], tokenize = False, add_generation_prompt = True)

    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature = 0.8,
        top_p = 0.95,
        max_tokens = 1024,
    )
    output = model.fast_generate(
        text,
        sampling_params = sampling_params,
        lora_request = model.load_lora("grpo_saved_lora"),
    )[0].outputs[0].text

    output

if __name__ == "__main__":
    fire.Fire(main)