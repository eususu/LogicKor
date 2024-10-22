import os
from typing import Optional
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_model(_model:str, lora_adapter:str, revision:str, output:str):
    device = torch.cuda.current_device()
    used_ram = torch.cuda.memory_allocated(device)

    triplet_path = f'{output}/triplet.json'

    class Triplet(BaseModel):
        model:str
        lora_adapter:str
        lora_adapter_revision:Optional[str]

    input_triplet:Triplet = Triplet(
        model=_model,
        lora_adapter=lora_adapter,
        lora_adapter_revision=revision,
    )
    triplet:Triplet = None

    if os.path.exists(output) and os.path.exists(triplet_path):
        print(f"already has merged model({output})")
        with open(triplet_path, 'r') as file:
            import json
            triplet_json = json.load(file)
            triplet = Triplet(**triplet_json)

        if triplet is not None:
            if triplet == input_triplet:
                return
            else:
                print("renew merging")
                print(f'saved triplet({triplet}), new triplet({input_triplet})')

    print(f"generating merge model to {output}")

    device_map = 'auto'

    with torch.no_grad():
        print(f'{_model} 을 로딩합니다.')
        model = AutoModelForCausalLM.from_pretrained(_model, device_map=device_map, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(_model, device_map=device_map)

        peft_args = {
            'torch_dtype': torch.bfloat16,
        }
        if revision is not None:
            peft_args['revision'] = revision

        print(f'lora_adapter={lora_adapter}, revision={revision} 을 로딩합니다.')
        model = PeftModel.from_pretrained(model, lora_adapter, **peft_args)

        print(f'병합 시작')
        model = model.merge_and_unload()
        print(f'병합결과를 기록합니다.')
        model.save_pretrained(output)
        tokenizer.save_pretrained(output)
        print(f'병합결과가 ({output})에 기록되었습니다.')

        del model
        del tokenizer

        with open(triplet_path, 'w') as file:
            file.write(input_triplet.model_dump_json())

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m", help=" : base model", required=True)
parser.add_argument("-lm", help=" : lora adapter", required=True)
parser.add_argument("-lmr", help=" : revision of lora adapter storage")
parser.add_argument("-o", help=" : output to saved", required=True)


if __name__ == "__main__":
    args = parser.parse_args()
    model = args.m
    lora_adapter = args.lm
    revision = args.lmr
    output = args.o

    merge_model(model, lora_adapter=lora_adapter, revision=revision, output=output)