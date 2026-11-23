import llm_sdk
import numpy as np
import sys
import os
import json

def load_json_file(path: str):
	with open(path, "r") as f:
		data = json.load(f)
	return data

def send_promet(
		promet: str,
        model: llm_sdk.Small_LLM_Model,
        vocap) -> str:
    TokenIDS = model.encode(promet).tolist()[0]
    STOP_TOKEN = 151645  #151645
    answer = []
    i = 0
    while i < 100:
        logits = model.get_logits_from_input_ids(TokenIDS)
        max_priority = int(np.argmax(logits))
        answer.append(max_priority)
        TokenIDS.append(max_priority)
        if STOP_TOKEN == max_priority:
            break
        i+=1
    return model.decode(answer)

def check_paths(functions_definition_file, input_file, output_file):
    if not(os.path.exists(functions_definition_file) and os.path.exists(input_file)):
        return 0
    try:
        parents_path = os.path.dirname(output_file)
        if parents_path:
            os.makedirs(name=parents_path, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump({}, f)
        return 1
    except Exception as e:
        print("Error: unexpected error in Creating the output file !!")
        return 0

def casting_with_ai(model, data, type_to):
    if type_to == "boolean":
        return bool(data)
    if type_to == "number":
        return int(data)
    else:
        return data

def determain_type(functions, func_name, parameter_name):
    for i in functions:
        if i["name"] == func_name:
            return i["parameters"][parameter_name]["type"]


def convert_to_map(string: str, promet, functions, model):
    split_str = string.split(",")
    split_str = [item.strip().strip("\n").replace("\"", "") for item in split_str]
    answer = {}
    answer.update({"prompt": promet})
    answer.update({"name": split_str[0]})
    tmep = {}
    for i in range(1, len(split_str)):
        a, b = split_str[i].split(":")
        b = b.strip()
        tmep.update({f"{a}": casting_with_ai(model=model, data=b, type_to=determain_type(functions, split_str[0], a))})
    answer.update({"parameters": tmep})
    return(answer)

def determain_the_function(model, functions, promet, vocap):
    model_resault = send_promet(
        promet = "<|im_start|>system\n"
        "You are very good at choosing a function , so i want your help to know what the function i must use."
        "\n i will give you some functions and for each function i will give you a description for it wich decsripe how the function work, "
        f"also i will send you tha paramters of each function and the return type of it and here are the functions:\n{functions}."
        "your job is to help me with detrmaine wich function to use by give me the name of the function and what is the value of each parameter that used in the function"
        "the output must be like this format:\nThe value of ths function name, the first parameter name:the value of it.\n"
        "and if there are more one paramter just ad (,) and then write (The Parameter name:the value of it).\nthis is some examples of the outuput format:\n"
        "fn_add_numbers, a:6, b:8\nfn_add_numbers, a:10, b:63\netc...\n"
        "be sure that it is forbidden the output contain any thing else!!"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{promet}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n",
        model=model,
        vocap=vocap)
    return(convert_to_map(model_resault, promet, functions, model))

def main():
    argv = sys.argv
    argv = argv[1:]
    if "--functions_definition" in argv:
        functions_definition_file = argv[argv.index("--functions_definition") + 1]
    else:
        functions_definition_file = "data/input/functions_definition.json"
    if "--input" in argv:
        input_file = argv[argv.index("--input") + 1]
    else:
        input_file = "data/input/function_calling_tests.json"
    if "--output" in argv:
        output_file = argv[argv.index("--output") + 1]
    else:
        output_file = "data/output/function_calls.json"
    #print(os.path.exists(output_file))
    if (not check_paths(functions_definition_file, input_file, output_file)):
        print("Error: Please enter a vailabe path !!")
        return
    try:
        with open(functions_definition_file) as f:
            functions = json.load(f)
    except Exception as e:
        print(e)
        return
    model = llm_sdk.Small_LLM_Model()
    answer = []
    vocap = load_json_file(model.get_path_to_vocab_file())
    with open(input_file, "r") as f:
        data = json.load(f)
        for i in data:
            answer.append(determain_the_function(model, functions, i["prompt"], vocap))
    with open(output_file, "w") as f:
        json.dump(answer, f)
    print(answer)

if __name__ == "__main__":
    import time
    
    start = time.perf_counter()
    main()
    elapsed = time.perf_counter() - start
    print(f"took {elapsed:.2f} seconds")