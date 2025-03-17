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
    elif one_shot == 1:
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
        
        ## Example
        
        Question:
        James decides to make a bathtub full of jello.  For every pound of water, you need 1.5 tablespoons of jello mix.  The bathtub can hold 6 cubic feet of water.  Each cubic foot of water is 7.5 gallons.  A gallon of water weighs 8 pounds.  A tablespoon of jello mix costs $0.50.  How much did he spend to fill his tub?
        
        <reasoning>
        Here you are reasoning
        </reasoning>
        
        <code>
        cost_to_fill_tub(X) :-
            TotalWater is 6 * 7.5,  % Total volume in gallons
            TotalWeight is TotalWater * 8, % Total weight in pounds
            TotalJelloMix is TotalWeight * 1.5,  % Total jello mix needed in tablespoons
            TotalCost is TotalJelloMix * 0.5, % Total cost in dollars
            X is TotalCost.
        </code>
        
        <query>
        cost_to_fill_tub(X).
        </query>
        """
    elif one_shot == 5:
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
        
        ## Example 1
        
        Question:
        James decides to make a bathtub full of jello.  For every pound of water, you need 1.5 tablespoons of jello mix.  The bathtub can hold 6 cubic feet of water.  Each cubic foot of water is 7.5 gallons.  A gallon of water weighs 8 pounds.  A tablespoon of jello mix costs $0.50.  How much did he spend to fill his tub?
        
        <reasoning>
        Here you are reasoning
        </reasoning>
        
        <code>
        cost_to_fill_tub(X) :-
            TotalWater is 6 * 7.5,  % Total volume in gallons
            TotalWeight is TotalWater * 8, % Total weight in pounds
            TotalJelloMix is TotalWeight * 1.5,  % Total jello mix needed in tablespoons
            TotalCost is TotalJelloMix * 0.5, % Total cost in dollars
            X is TotalCost.
        </code>
        
        <query>
        cost_to_fill_tub(X).
        </query>


        ## Example 2

        Question:
        Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a budget of $200 and spent $30 on a button-up shirt, $46 on suit pants, $38 on a suit coat, $11 on socks, and $18 on a belt. She also purchased a pair of shoes, but lost the receipt for them. She has $16 left from her budget. How much did Alexis pay for the shoes?

        <reasoning>
        Here you are reasoning
        </reasoning>

        <code>
        budget(alexis, 200).
        shirt_price(30).
        suit_pants_price(46).
        suit_coat_price(38).
        socks_price(11).
        belt_price(18).
        left(16).
        solve(Shoes_price) :-
            budget(alexis, Budget),
            shirt_price(Shirt_price),
            suit_pants_price(Suit_pants_price),
            suit_coat_price(Suit_coat_price),
            socks_price(Socks_price),
            belt_price(Belt_price),
            left(Left),
            {Budget = Shirt_price + Suit_pants_price + Suit_coat_price + Socks_price + Belt_price + Shoes_price + Left}.
        </code>

        <query>
        solve(Shoes_price).
        </query>


        ## Example 3

        Question:
        Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?

        <reasoning>
        Here you are reasoning
        </reasoning>

        <code>
        trees(randy, mango, 60).
        solve(Total_trees) :-
            trees(randy, mango, Mango_trees),
            {Half_mango_trees = Mango_trees / 2},
            {Coconut_trees = Half_mango_trees - 5},
            {Total_trees = Mango_trees + Coconut_trees}.
        </code>

        <query>
        solve(Total_trees).
        </query>


        ## Example 4

        Question:
        A car is driving through a tunnel with many turns. After a while, the car must travel through a ring that requires a total of 4 right-hand turns. After the 1st turn, it travels 5 meters. After the 2nd turn, it travels 8 meters. After the 3rd turn, it travels a little further and at the 4th turn, it immediately exits the tunnel. If the car has driven a total of 23 meters around the ring, how far did it have to travel after the 3rd turn?

        <reasoning>
        Here you are reasoning
        </reasoning>

        <code>
        right_turns(car, 4).
        turn(car, 1, 5).
        turn(car, 2, 8).
        turn(car, 4, 0).
        total_distance(car, 23).
        solve(Distance_after_3rd_turn) :-
            turn(car, 1, Distance_at_1st_turn),
            turn(car, 2, Distance_at_2nd_turn),
            turn(car, 4, Distance_at_4th_turn),
            total_distance(car, Total_distance),
            {Total_distance = Distance_at_1st_turn + Distance_at_2nd_turn + Distance_after_3rd_turn + Distance_at_4th_turn}.
        </code>

        <query>
        solve(Distance_after_3rd_turn).
        </query>


        ## Example 5

        Question:
        Leo's assignment was divided into three parts. He finished the first part of his assignment in 25 minutes. It took him twice as long to finish the second part. If he was able to finish his assignment in 2 hours, how many minutes did Leo finish the third part of the assignment?

        <reasoning>
        Here you are reasoning
        </reasoning>

        <code>
        assignment_part_time(leo, part1, 25).
        assignment_total_time(leo, 120).
        solve(Part3_time) :-
            assignment_part_time(leo, part1, Part1_time),
            assignment_total_time(leo, Total_time),
            {Part1_time + Part2_time + Part3_time = Total_time},
            {Part2_time = 2 * Part1_time}.
        </code>

        <query>
        solve(Part3_time).
        </query>
        """

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
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

    os.environ["WANDB_PROJECT"] = "prolog-3b"
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    os.environ["WANDB_ENTITY"] = "halykoss"


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
        from pyswip import Prolog
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
            # Use the instance's query method
            result = list(prolog_interpreter.query(query))
            for inference in result:
                for _, result_inference in inference.items():
                    print("Infered: {}, Response: {}, Match: {}".format(result_inference, answer, float(result_inference) == float(answer)))
                    try:
                        if float(result_inference) == float(answer):
                            return 1
                    except:
                        print("Matching error!")
                        return -0.5
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
            reward_achieved = run_with_timeout(knowledge_base, query, a)
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
            # if "?-" in text between <code> and </code> then reward -0.5
            code_text = text.split("<code>\n")[-1]
            code_text = code_text.split("\n</code>")[0]
            query_text = text.split("<query>\n")[-1]
            query_text = query_text.split("\n</query>")[0]
            if "?-" in code_text or query_text.replace("\n", "") in code_text:
                print("\nFOUND QUERY IN CODE: ", query_text.replace("\n", ""))
                count -= 0.5
        if text.count("\n</code>\n") == 1:
            count += 0.125
        if text.count("\n<query>\n") == 1:
            count += 0.125
            count -= len(text.split("\n</query>\n")[-1])*0.001
        if text.count("\n</query>") == 1:
            count += 0.125
            count -= (len(text.split("\n</query>")[-1]) - 1)*0.001
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
        # num_train_epochs = 1, # Set to 1 for a full training run
        max_steps = 1500,
        save_steps = 500,
        max_grad_norm = 0.1,
        report_to = "wandb", # Can use Weights & Biases
        output_dir="Qwen-"+model_B+"B"+shot_string,
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