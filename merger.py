import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_model(_model:str, lora_adapter:str, revision:str, output:str):

    if os.path.exists(output):
        print(f"already has merged model({output})")
        return

    print(f"generating merge model to {output}")


    device_map = 'auto'

    with torch.no_rad():
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
        print(f'병합이 완료되었습니다.')
        model.save_pretrained(output)
        tokenizer.save_pretrained(output)
        print(f'병합결과가 {output}에 기록되었습니다.')

        torch.cuda.empty_cache()