import os
import warnings
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import random
import json
from captum.attr import (
    FeatureAblation, 
    ShapleyValues,
    LayerIntegratedGradients, 
    LLMAttribution, 
    LLMGradientAttribution, 
    TextTokenInput, 
    TextTemplateInput,
    ProductBaselines,
)

# !!! set for CINECA GPU
os.environ["CUDA_VISIBLE_DEVICES"]="5"

# Ignore warnings due to transformers library
warnings.filterwarnings("ignore", ".*past_key_values.*")
warnings.filterwarnings("ignore", ".*Skipping this token.*")

# set seed
torch.manual_seed(0)
torch.cuda.manual_seed(0)
random.seed(0)
transformers.set_seed(0)


def load_model(model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        do_sample=False # Set to False to always generate the same output
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
    # Needed for LLaMA tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def preprocess_data(path):
    questions = None
    answers = None
    with open(path, "r") as file:
        data = [json.loads(line) for line in file]
        questions = [x["question"] for x in data]
        # answers are in the following format
        # "answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"
        # we need to extract the last number from the string after the "####" separator
        answers = [x["answer"].split("####")[1].strip() for x in data]
    return questions, answers


def create_prompt(question, is_json_format):
    prefix = 'You are a math tutor who helps students of all levels understand and solve mathematical problems. Read the "Question" carefully and answer. The final answer must be only a number. "Question": '
    suffix = ' Provide your output in the following valid text format: ``` answer: ... ```. Response: '
    suffix_json = ' Provide your output in the following valid JSON format: ``` {{ "answer": ... }} ```. Response: '

    prompt = prefix + question
    prompt += suffix_json if is_json_format else suffix

    # Split the prompt into tokens
    splitted_prompt = prompt.split(" ")
    # Create a list of tokens to be analyzed, in this case all tokens in the prompt
    # Change the following line to select a subset of tokens to be analyzed instead of splitted_prompt[:]
    tokens_to_analyze = splitted_prompt[:]

    # Create a new prompt with placeholders {} for the tokens to be analyzed, in this case all tokens in the prompt.
    # In this case the new prompt will be a string with the same length as the original prompt, but with all tokens replaced by {}
    prompt_with_placeholders = prompt
    for token in tokens_to_analyze:
        prompt_with_placeholders = prompt_with_placeholders.replace(token, "{}", 1)

    return prompt, prompt_with_placeholders, tokens_to_analyze


def features_plot(dir_path, model, tokenizer, prompt_with_placeholders, tokens_to_analyze, answer, is_json_format):
    print("Generating features plot...")
    skip_tokens = [1]

    fa = FeatureAblation(model)
    llm_attr = LLMAttribution(fa, tokenizer)

    inp = TextTemplateInput(
        template=prompt_with_placeholders,
        values=tokens_to_analyze
    )

    target = None
    if is_json_format:
        target = '``` { "answer": ' + str(answer) + ' } ```'
    else:
        target = "``` answer: " + str(answer) + " ```"
    
    attr_res = llm_attr.attribute(inp, target=target, skip_tokens=skip_tokens)
    fig, _ = attr_res.plot_token_attr()

    # save the plot as pdf
    dir_path += "features/"
    file_path = dir_path
    file_path += "json.pdf" if is_json_format else "text.pdf"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    fig.savefig(file_path, dpi=300)


def gradient_plot(dir_path, model, tokenizer, prompt, answer, is_json_format):
    print("Generating gradient plot...")
    lig = LayerIntegratedGradients(model, model.model.embed_tokens)
    llm_attr = LLMGradientAttribution(lig, tokenizer)

    inp = TextTokenInput(
        text = prompt,
        tokenizer = tokenizer
    )

    target = None
    if is_json_format:
        target = '``` { "answer": ' + str(answer) + ' } ```'
    else:
        target = "``` answer: " + str(answer) + " ```"
    
    attr_res = llm_attr.attribute(inp, target=target)
    fig, _ = attr_res.plot_token_attr()

    # save the plot as pdf
    dir_path += "gradient/"
    file_path = dir_path
    file_path += "json.pdf" if is_json_format else "text.pdf"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    fig.savefig(file_path, dpi=300)



def generate_plots(dir_path, model, tokenizer, prompt, prompt_with_placeholders, tokens_to_analyze, answer, is_json_format):
    features_plot(dir_path, model, tokenizer, prompt_with_placeholders, tokens_to_analyze, answer, is_json_format)
    gradient_plot(dir_path, model, tokenizer, prompt, answer, is_json_format)




model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model, tokenizer = load_model(model_name)

questions, answers = preprocess_data("data/train.jsonl")

# !!!!!! Here create a for loop to iterate over all questions

question_id = 0
print("\n")
print("questions[0]:", questions[question_id])
print("answers[0]:", answers[question_id])

# Test for json format
is_json_format = True
prompt, prompt_with_placeholders, tokens_to_analyze = create_prompt(questions[question_id], is_json_format=is_json_format)
print("\n")
print("prompt:", prompt)
print("\n")
print("prompt_with_placeholders:", prompt_with_placeholders)
print("\n")
print("tokens_to_analyze:", tokens_to_analyze)
generate_plots("plots/", model, tokenizer, prompt, prompt_with_placeholders, tokens_to_analyze, answers[question_id], is_json_format=is_json_format)

""" 
# Test for text format
is_json_format = False
prompt, prompt_with_placeholders, tokens_to_analyze = create_prompt(questions[question_id], is_json_format=is_json_format)
print("\n")
print("prompt:", prompt)
print("\n")
print("prompt_with_placeholders:", prompt_with_placeholders)
print("\n")
print("tokens_to_analyze:", tokens_to_analyze)
generate_plots("plots/", model, tokenizer, prompt, prompt_with_placeholders, tokens_to_analyze, answers[question_id], is_json_format=is_json_format)
"""