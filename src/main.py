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
        model: llm_sdk.Small_LLM_Model) -> str:
    TokenIDS = model.encode(promet).tolist()[0]
    vocap = load_json_file(model.get_path_to_vocab_file())
    STOP_TOKEN = 151643  #151645
    answer = []
    i = 0
    while i < 300:
        print(i)
        logits = np.array(model.get_logits_from_input_ids(TokenIDS))
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
    print()
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
    print(functions_definition_file)
    print(input_file)
    print(output_file)
    model = llm_sdk.Small_LLM_Model()
    encode_str = model.encode("Hi my name is Rashed")
    print(send_promet(
        promet = "<|im_start|>system\n"
        "you are mathematician and very good in maths so answer any math equation in simple answer"
        "\n don't forget there are negative numbers like -4 and -7 etc ... and in a - b equation if b > a the answer is negative number"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "7 / 2"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>i\n\n",
        model=model))    

if __name__ == "__main__":
    main()
