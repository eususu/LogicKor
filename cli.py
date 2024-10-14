import argparse
import os
import subprocess
parser = argparse.ArgumentParser()


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
    model = args.m
    lora_adapter = args.lm

    if os.environ["OPENAI_API_KEY"]:
        print("use apikey in ENV")
        apikey = os.environ["OPENAI_API_KEY"]
    else:
        apikey = args.k

    
    if apikey is None:
        raise ValueError("need openai apikey (use option -k or environment variable)")
    print(f"go eval model({model}), lora_adapter({lora_adapter})")

    if args.force:
        # 디렉토리 및 파일 모두 삭제
        import shutil
        shutil.rmtree(f'./generated/{model}', ignore_errors=True)
        shutil.rmtree('./evaluated/default.jsonl', ignore_errors=True)
        shutil.rmtree('./evaluated/cot-1-shot.jsonl', ignore_errors=True)
        shutil.rmtree('./evaluated/1-shot.jsonl', ignore_errors=True)
    cmd = [
        'python3',
        'generator.py',
        '-m',
        model,
    ]

    if lora_adapter:
        cmd.append('-lora')
        cmd.append(lora_adapter)
    subprocess.call(cmd)

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
    subprocess.call(cmd)

    score_command(args)





subparsers = parser.add_subparsers(title='commands', dest='command')

clear_parser = subparsers.add_parser('clear', help='생성된 파일을 삭제합니다.')
clear_parser.set_defaults(func=clear_command)
score_parser = subparsers.add_parser('score', help='평가 점수를 계산합니다')
score_parser.set_defaults(func=score_command)
eval_parser = subparsers.add_parser('eval', help='주어진 모델에 대한 평가를 수행합니다.')
eval_parser.add_argument("-m", type=str, help="평가할 모델의 경로 또는 huggingface 경로(ex: aiyets/gemma-2-9b-it-dpo)", required=True)
eval_parser.add_argument("-lm", type=str, help="평가할 LORA 모델의 경로 또는 huggingface 경로(ex: aiyets/gemma-2-9b-it-dpo-lora)")
eval_parser.add_argument("-k", type=str, help="OPENAI API KEY. 환경 변수에 있으면, 이 옵션에 전달하지 않아도 됨")
eval_parser.add_argument("-f", dest='force', action='store_true', help="이미 평가된 파일이 있으면 삭제하고 새로 생성합니다.")
eval_parser.set_defaults(func=eval_command)


args = parser.parse_args()
if args.command is None:
    parser.print_help()
    exit(-1)

print(args)
args.func(args)


"""
subprocess.call()

python3 generator.py -m google/gemma-2-2b-it -lora ../ai/vrl/simpo_result/

python3 evaluator.py -o ./generated/google/gemma-2-2b-it/default.jsonl -k ${OPENAI_API_KEY} -j gpt-4o -t 30

python3 score.py -p ./evaluated/default.jsonl
"""