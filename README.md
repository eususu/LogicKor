# LogicKor

한국어 언어모델 다분야 사고력 벤치마크

## Benchmark Website

<https://lk.instruct.kr>

## Note

pr 적극 환영합니다.
벤치마크 결과 Self-Report도 받습니다. issue나 pr 부탁드립니다. 💕
* 권장 사항: PR 이전에 `make format && make check` 를 통해 코드 포맷팅을 확인해주세요. (black, isort, ruff 의존성 설치 필요)

## Repository

본 Repo는 LogicKor 벤치마크의 추론 및 평가 코드, 데이터셋을 담고 있습니다.

## Evaluation Example

GPU 0,1 사용, model_len 4096

### 1. 인퍼런스 결과 생성

```bash
python generator.py --model yanolja/EEVE-Korean-Instruct-10.8B-v1.0 --gpu_devices 0,1 --model_len 4096
```

### 2. Judge 모델로 평가

#### OpenAI

```bash
python evaluator.py -o ./generated/yanolja/EEVE-Korean-Instruct-10.8B-v1.0 -k sk-somethingsomething -t 30
```

#### Azure

```bash
export AZURE_ENDPOINT=$AZURE_ENDPOINT
export AZURE_DEPLOYMENT_NAME=$AZURE_DEPLOYMENT_NAME
export AZURE_API_VERSION=$AZURE_API_VERSION

python evaluator.py --azure -o ./generated/yanolja/EEVE-Korean-Instruct-10.8B-v1.0 -k sk-somethingsomething -t 30
```

### 3. 결과 확인

```bash
python score.py -p ./evaluated/yanolja/EEVE-Korean-Instruct-10.8B-v1.0/default.jsonl
```


### 통합 사용

| 명령 | 옵션 |
| --- | --- |
| eval | -m MODEL_NAME<br/>-lm LORA_ADAPTER_PATH <br/>-f FORCE OVERWRITE |
| clear | X |

```bash
python3 cli.py eval -m google/gemma-2-2b-it -lm ../ai/vrl/dpo_result -f
```

`결과`
```bash
go eval model(google/gemma-2-2b-it), lora_adapter(../ai/vrl/dpo_result)
- Using vLLM
Args - Namespace(gpu_devices='0', model='google/gemma-2-2b-it', model_len=4096, lora_module='../ai/vrl/dpo_result')
INFO 10-14 17:01:29 llm_engine.py:223] Initializing an LLM engine (v0.6.1.post2) with config: model='google/gemma-2-2b-it', speculative_config=None, tokenizer='google/gemma-2-2b-it', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, rope_scaling=None, rope_theta=None, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=4096, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=google/gemma-2-2b-it, use_v2_block_manager=False, num_scheduler_steps=1, enable_prefix_caching=False, use_async_output_proc=True)
INFO 10-14 17:01:30 model_runner.py:997] Starting to load model google/gemma-2-2b-it...
INFO 10-14 17:01:31 weight_utils.py:242] Using model weights format ['*.safetensors']
Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:03<00:00,  1.64s/it]

... 중략 ...

./generated/google/gemma-2-2b-it
Found 3 JSON files to process
- 현재 Processing : generated/google/gemma-2-2b-it/default.jsonl
- 현재 Processing : generated/google/gemma-2-2b-it/1-shot.jsonl
- 현재 Processing : generated/google/gemma-2-2b-it/cot-1-shot.jsonl
| 글쓰기(Writing) | 문법(Grammar) | 수학(Math) | 이해(Understanding) | 추론(Reasoning) | 코딩(Coding) | Single turn | Multi turn | Overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.43, 5.29 | 5.00, 2.86 | 4.86, 4.86 | 7.71, 7.57 | 6.43, 3.43 | 6.43, 8.29 | 5.81 | 5.38 | 5.60 |
```
