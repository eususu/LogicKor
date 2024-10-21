import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_model(_model:str, lora_adapter:str, output:str):

    if os.path.exists(output):
        print(f"already has merged model({output})")
        return

    print(f"generating merge model to {output}")


    device_map = 'auto'

    model = AutoModelForCausalLM.from_pretrained(_model, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(_model, device_map=device_map)
    model = PeftModel.from_pretrained(model, lora_adapter)
    model = model.merge_and_unload()
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)