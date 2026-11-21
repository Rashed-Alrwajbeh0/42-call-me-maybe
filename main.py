import llm_sdk
import numpy as np
import bisect
import json

def load_json_file(path: str):
	with open(path, "r") as f:
		data = json.load(f)
	return data

def send_promet(
		promet: str,
        model: llm_sdk.Small_LLM_Model) -> list[int]:
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

def main():
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
