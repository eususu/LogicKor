import os
from typing import Optional
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_model(_model:str, lora_adapter:str, revision:str, output:str):
    class Triplet(BaseModel):
        model:str
        lora_adapter:str
        lora_adapter_revision:Optional[str]

    input_triplet:Triplet = {
        'model': _model,
        'lora_adapter': lora_adapter,
        'lora_adapter_revision': revision,
    }
    triplet:Triplet = None

    if os.path.exists(output):
        print(f"already has merged model({output})")
        with open('triplet.json', 'r') as file:
            triplet = Triplet.model_validate(file)

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

        print(f'{lora_adapter} revision={revision} 을 로딩합니다.')
        model = PeftModel.from_pretrained(model, lora_adapter, **peft_args)

        print(f'병합 시작')
        model = model.merge_and_unload()
        print(f'병합결과를 기록합니다.')
        model.save_pretrained(output)
        tokenizer.save_pretrained(output)
        print(f'병합결과가 ({output})에 기록되었습니다.')

        with open('triplet.json', 'w') as file:
            file.write(triplet.model_dump())

    torch.cuda.empty_cache()