import argparse
import os
import subprocess
import shutil
import torch

parser = argparse.ArgumentParser()



if not torch.cuda.is_available():
    raise Exception("CUDA is not available!!!")

def clear_command(args):
    print('go clear')

def score_command(args):

    files = ['default.jsonl', 'cot-1-shot.jsonl', '1-shot.jsonl']

    for file in files:
        cmd = [
        'python3',
        'score.py',
        '-p',
        f'./evaluated/{file}',
        ]
        print(f'RESULT({file})')
        subprocess.call(cmd)

def eval_command(args:argparse.Namespace):
    device_count=torch.cuda.device_count()
    devices = [str(count) for count in range(device_count)]
    print(f"detected CUDA DEVICES: ({devices})")

    model = args.m
    lora_adapter = args.lm
    lora_adapter_revision = args.lmr

    if os.environ["OPENAI_API_KEY"]:
        print("use apikey in ENV")
        apikey = os.environ["OPENAI_API_KEY"]
    else:
        apikey = args.k
    
    if apikey is None:
        raise ValueError("need openai apikey (use option -k or environment variable)")
    print(f"GO eval model({model}), lora_adapter({lora_adapter})")

    merged_model = f"{model}_auto_merged"
    if args.force:
        remove_list = [
        f'./generated/{model}',
        f'./generated/{merged_model}',
        './evaluated/default.jsonl',
        './evaluated/cot-1-shot.jsonl',
        './evaluated/1-shot.jsonl',
        ]
        # 디렉토리 및 파일 모두 삭제
        import shutil
        for f in remove_list:
            print(f'# {f} 파일(디렉터리)을 삭제합니다')
            if os.path.isfile(f):
                os.remove(f)
            else:
                shutil.rmtree(f, ignore_errors=True)
    
    if lora_adapter:
        # merge: vllm이 lora rank 16이상은 지원안하므로, 합쳐서 평가함

        cmd = [
        'python3',
        'merger.py',
        '-m',
        model,
        '-o',
        merged_model,
        '-lm',
        lora_adapter
        ]

        if lora_adapter_revision is not None:
            cmd.append('-lmr')
            cmd.append(lora_adapter_revision)

        exit_code = subprocess.call(cmd)
        if exit_code != 0:
            print(f'failed subprocess job(exit_code)')
            exit(exit_code)
        
        # redirect to merged model
        model = merged_model

    cmd = [
        'python3',
        'generator.py',
        '-m',
        model,
        '-g',
        ",".join(devices)
    ]
    print(cmd)
    exit_code = subprocess.call(cmd)
    if exit_code != 0:
        print(f'failed subprocess job(exit_code)')
        exit(exit_code)

    model_output_dir= f'./generated/{model}'
    print(model_output_dir)
    cmd = [
        'python3',
        'evaluator.py',
        '-o',
        model_output_dir,
        '-k',
        apikey,
        '-j',
        'gpt-4o',
        '-t',
        '30'
    ]
    print(cmd)
    exit_code = subprocess.call(cmd)
    if exit_code != 0:
        print(f'failed subprocess job(exit_code)')
        exit(exit_code)

    # 평가 파일 복사
    if os.path.exists(model):
        # 머징된 경우에 추가 해줌
        dest_path = f'{model}/LogicKor'
        if not os.path.exists(dest_path):
            os.mkdir(dest_path)
        copy_list = [
            'default.jsonl',
            'cot-1-shot.jsonl',
            '1-shot.jsonl',
        ]
        for file in copy_list:
            shutil.copyfile(f'./evaluated/{file}', f'{dest_path}/{file}')

    score_command(args)

subparsers = parser.add_subparsers(title='commands', dest='command')

clear_parser = subparsers.add_parser('clear', help='생성된 파일을 삭제합니다.')
clear_parser.set_defaults(func=clear_command)
score_parser = subparsers.add_parser('score', help='평가 점수를 계산합니다')
score_parser.set_defaults(func=score_command)
eval_parser = subparsers.add_parser('eval', help='주어진 모델에 대한 평가를 수행합니다.')
eval_parser.add_argument("-m", type=str, help="평가할 모델의 경로 또는 huggingface 경로(ex: aiyets/gemma-2-9b-it-dpo)", required=True)
eval_parser.add_argument("-lm", type=str, help="평가할 LORA 모델의 경로 또는 huggingface 경로(ex: aiyets/gemma-2-9b-it-dpo-lora)")
eval_parser.add_argument("-lmr", type=str, help="평가할 LORA 모델의 git tag, commit hash(ex: 83985f284975e9a08edceb56982a4fea7b021139)")
eval_parser.add_argument("-k", type=str, help="OPENAI API KEY. 환경 변수에 있으면, 이 옵션에 전달하지 않아도 됨")
eval_parser.add_argument("-f", dest='force', action='store_true', help="이미 평가된 파일이 있으면 삭제하고 새로 생성합니다.")
eval_parser.set_defaults(func=eval_command)


args = parser.parse_args()
if args.command is None:
    parser.print_help()
    exit(-1)

print(args)
args.func(args)
