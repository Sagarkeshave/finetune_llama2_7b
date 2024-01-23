import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel



def infer_lm():
    # base_model
    model_name = "NousResearch/Llama-2-7b-chat-hf"

    # Fine-tuned model name
    ft_model = "SagarKeshave/main"

    # Load the entire model on the GPU 0
    device_map = {"": 0}

    # Reload model in FP16 and merge it with LoRA weights
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(base_model, ft_model)
    model = model.merge_and_unload()

    # Reload tokenizer to save it
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Enter a prompt to generate text or type 'exit' to quit.")
    while True:
        prompt = input("Prompt: ")

        if prompt.lower() == "exit":
            break

        input_tokens = tokenizer.encode(prompt, return_tensors="pt")
        output_tokens = model.generate(input_tokens, max_length=20, num_return_sequences=1)
        output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        print(f"Generated text: {output_text}")


infer_lm()