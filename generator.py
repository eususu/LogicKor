import argparse
import os

import pandas as pd

from templates import PROMPT_STRATEGY

# Use aphrodite-engine or vLLM
try:
    from aphrodite import LLM, SamplingParams

    print("- Using aphrodite-engine")

except ImportError:
    from vllm import LLM, SamplingParams 
    from vllm.lora.request import LoRARequest

    print("- Using vLLM")

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu_devices", help=" : CUDA_VISIBLE_DEVICES", default="0")
parser.add_argument(
    "-m",
    "--model",
    help=" : Model to evaluate",
    default="yanolja/EEVE-Korean-Instruct-2.8B-v1.0",
)
parser.add_argument("-ml", "--model_len", help=" : Maximum Model Length", default=4096, type=int)
parser.add_argument("-lora", "--lora_module", help=" : Lora Module Path", default='', type=str)
args = parser.parse_args()

print(f"Args - {args}")

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
gpu_counts = len(args.gpu_devices.split(","))

llm = LLM(
    model=args.model,
    tensor_parallel_size=gpu_counts,
    max_model_len=args.model_len,
    gpu_memory_utilization=0.8,
    trust_remote_code=True,  # !
    enable_lora=False if len(args.lora_module) == 0 else True,

)

# detect gemma2 engine
NONE_SYSTEM_ROLE_CONFIGS = [
    "Gemma2Config"
]
_internal_llm_config_type = type(llm.llm_engine.model_config.hf_config).__name__
print(f'Detected LLM Config is ({_internal_llm_config_type})')
if _internal_llm_config_type in NONE_SYSTEM_ROLE_CONFIGS:
    support_system_role = False
else:
    support_system_role = True
print("System role using = ", support_system_role)

sampling_params = SamplingParams(
    temperature=0,
    skip_special_tokens=True,
    max_tokens=args.model_len,
    stop=["<|endoftext|>", "[INST]", "[/INST]", "<|im_end|>", "<|end|>", "<|eot_id|>", "<end_of_turn>", "<eos>"],
)

df_questions = pd.read_json("questions.jsonl", orient="records", encoding="utf-8-sig", lines=True)

if not os.path.exists("./generated/" + args.model):
    os.makedirs("./generated/" + args.model)


for strategy_name, prompts in PROMPT_STRATEGY.items():
    print(prompts)

    if not support_system_role:
        # merge content until first user role
        contents = []
        for index, prompt in enumerate(prompts):
            if prompt["role"] == "system":
                contents.append(prompt["content"])
            else:
                contents.append(prompt["content"])
                prompt["content"] = "\n".join(contents) # merge the accumulated contents to user's content
                prompts = prompts[index::] # remove system role
                break
        print(prompt)

    def format_single_turn_question(question):
        return llm.llm_engine.tokenizer.tokenizer.apply_chat_template(
            prompts + [{"role": "user", "content": question[0]}],
            tokenize=False,
            add_generation_prompt=True,
        )

    single_turn_questions = df_questions["questions"].map(format_single_turn_question)
    print(single_turn_questions.iloc[0])
    single_turn_outputs = [
        output.outputs[0].text.strip() for output in llm.generate(single_turn_questions, sampling_params, lora_request=None if len(args.lora_module) == 0 else LoRARequest('lora', 1, args.lora_module),)
    ]

    def format_double_turn_question(question, single_turn_output):
        return llm.llm_engine.tokenizer.tokenizer.apply_chat_template(
            prompts
            + [
                {"role": "user", "content": question[0]},
                {"role": "assistant", "content": single_turn_output},
                {"role": "user", "content": question[1]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    multi_turn_questions = df_questions[["questions", "id"]].apply(
        lambda x: format_double_turn_question(x["questions"], single_turn_outputs[x["id"] - 1]),
        axis=1,
    )
    multi_turn_outputs = [
        output.outputs[0].text.strip() for output in llm.generate(multi_turn_questions, sampling_params, lora_request=None if len(args.lora_module) == 0 else LoRARequest('lora', 1, args.lora_module),)
    ]

    df_output = pd.DataFrame(
        {
            "id": df_questions["id"],
            "category": df_questions["category"],
            "questions": df_questions["questions"],
            "outputs": list(zip(single_turn_outputs, multi_turn_outputs)),
            "references": df_questions["references"],
        }
    )
    df_output.to_json(
        "./generated/" + os.path.join(args.model, f"{strategy_name}.jsonl"),
        orient="records",
        lines=True,
        force_ascii=False,
    )
