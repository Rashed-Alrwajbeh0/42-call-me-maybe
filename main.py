import llm_sdk


def main():
	model = llm_sdk.Small_LLM_Model()
	print("Hello from call-me-mybe!")
	encode_str = model.encode("Hi my name is Rashed")
	print(model.get_logits_from_input_ids(encode_str.tolist()[0]))


if __name__ == "__main__":
    main()
