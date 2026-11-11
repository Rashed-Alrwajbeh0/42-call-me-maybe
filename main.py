import llm_sdk
import numpy as np
import bisect
import json

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def inverse_softmax(probabilities):
    return np.log(probabilities)

def find_key(sorted_items: dict, target):
    values = list(sorted_items.values())
    index = bisect.bisect_left(values, target)
    if index < len(values) and values[index] == target:
        return sorted_items[index][0]
    return None

def load_json_file(path: str):
	with open(path, "r") as f:
		data = json.load(f)
	return data

def generate_ans(
		model: llm_sdk.Small_LLM_Model,
		logits: np.array,
		TokensID: list[int],
		vocap: dict):
	STOP_TOKEN = 151645
	answer = []
	o = 1
	while 1:
		# print(o)
		# o+=1
		logits_after_softmax = softmax(logits)
		max_priority = int(np.argmax(logits_after_softmax))
		# print(max_priority)
		if softmax(STOP_TOKEN) == max_priority:
			break
		# The error is here because i must set the index to the vocabulary not in the TokensID list
		# answer.append(find_key(vocap, max_priority))
		# print(inverse_softmax(max_priority))
		print(list(vocap.values())[max_priority])
	return 1
		


def send_promet(
		promet: str,
        model: llm_sdk.Small_LLM_Model) -> list[int]:
	TokenIDS = model.encode(promet).tolist()[0]
	logits = np.array(model.get_logits_from_input_ids(TokenIDS))
	vocap = load_json_file(model.get_path_to_vocab_file())
	return generate_ans(model=model, logits=logits, TokensID=TokenIDS, vocap=vocap)

def main():
	model = llm_sdk.Small_LLM_Model()
	print("Hello from call-me-mybe!")
	encode_str = model.encode("Hi my name is Rashed")
	print(send_promet(promet="Hi my name is rashed", model=model))


if __name__ == "__main__":
    main()
