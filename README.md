# RL Learned Plan Search - GCP Ablation

## Quick Start

### 1. Docker 컨테이너 실행

```bash
sudo docker run -d --name verl --gpus all -p 8000:8000 --shm-size=64g \
  -v /home/robin:/home/robin -v /ckpt:/ckpt \
  -it verlai/verl:app-verl0.6-transformers4.56.1-sglang0.5.2-mcore0.13.0-te2.2 \
  /bin/bash -c "sleep infinity"
```

### 2. 컨테이너 접속

```bash
sudo docker exec -it verl /bin/bash
```

### 3. Conda 설치 및 환경 설정 (컨테이너 내부)

```bash
# Miniconda 설치
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p /opt/conda
rm /tmp/miniconda.sh

# Conda 초기화
source /opt/conda/etc/profile.d/conda.sh

# Terms of Service 동의
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# retrieval 환경 생성 (Python 3.12 + faiss-gpu)
conda create -n retrieval python=3.12 -y
conda activate retrieval
conda install -c pytorch -c nvidia faiss-gpu=1.8.0 -y
pip install datasets transformers uvicorn fastapi pydantic tqdm torch
```

### 4. Retrieval Server 실행 (컨테이너 내부)

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate retrieval
cd /home/robin/rl-learned-plan-search

nohup python examples/sglang_multiturn/search_r1_like/local_dense_retriever/retrieval_server.py \
  --index_path /ckpt/retrieval_data/e5_Flat.index \
  --corpus_path /ckpt/retrieval_data/wiki-18.jsonl \
  --retriever_name e5 \
  --retriever_model intfloat/e5-base-v2 \
  --topk 3 \
  --faiss_gpu > retrieval.log 2>&1 &
```

> **Note**: 인덱스 로딩에 약 2-3분 소요됩니다. `tail -f retrieval.log`로 진행 상황을 확인하세요.

### 5. Retrieval Server 테스트

```bash
curl -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"queries": ["What is Python?"], "topk": 3, "return_scores": true}'
```

### 6. 학습 실행 (컨테이너 내부)

```bash
cd /home/robin/rl-learned-plan-search

nohup bash examples/sglang_multiturn/search_r1_like/run_qwen3-1_7b_search_w_selective_think_entropy.sh \
  > train.log 2>&1 &
```

로그 확인:
```bash
tail -f train.log
```

---

## 주요 설정 파일

- **학습 스크립트**: `examples/sglang_multiturn/search_r1_like/run_qwen3-1_7b_search_w_selective_think_entropy.sh`
- **Tool Config**: `examples/sglang_multiturn/config/tool_config/search_tool_config_with_thinking_wo_goal.yaml`
- **Hydra Config**: `examples/sglang_multiturn/config/search_multiturn_grpo.yaml`

## 데이터 경로

- **Index**: `/ckpt/retrieval_data/e5_Flat.index` (64GB)
- **Corpus**: `/ckpt/retrieval_data/wiki-18.jsonl` (14GB)
- **Train Data**: `$HOME/data/searchR1_processed_w_selective_plan/train.parquet`
- **Val Data**: `$HOME/data/searchR1_processed_w_selective_plan/test.parquet`

## Troubleshooting

### NCCL shared memory 에러
Docker 실행 시 `--shm-size=64g` 옵션이 빠졌는지 확인하세요.

### FlashAttention 에러
H200 GPU에서는 `attention_backend=flashinfer` 설정이 필요합니다. 이미 스크립트에 포함되어 있습니다.

### Retrieval Server timeout
faiss-gpu가 제대로 설치되었는지 확인하세요. CPU 버전은 검색당 10초 이상 걸립니다.
