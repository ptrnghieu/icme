#!/usr/bin/env bash
# P17 on a rented GPU (vast.ai). Run inside tmux so an SSH disconnect does not stop it:
#   tmux new -s p17
#   export HF_TOKEN=...            # never paste it anywhere else; MER2026 dataset access is required
#   export MER2026_ZIP_PW=...      # only if the script says the zip is encrypted
#   bash run_p17.sh setup          # once: system packages, two Python environments, GPU / bf16 / vLLM checks
#   bash run_p17.sh prepare        # data: MER2026 human zip, MELD train clips, items.jsonl
#   bash run_p17.sh timing         # about 20 clips: seconds per clip and the structure of the descriptions
#   bash run_p17.sh describe       # phase 1 (resumable): all clips, 1 sample seed
#   bash run_p17.sh describe_seeds # optional, later: OV-MERD+ sample seeds 2 and 3
#   bash run_p17.sh extract        # phase 2 (resumable): official extractor on all descriptions
#   bash run_p17.sh pack           # p17_out.tar.gz: descriptions and labels only (no media, no annotations)
set -euo pipefail
W=${W:-/workspace/p17}; REPO=$W/icme; OV=$REPO/explore/ovmer; mkdir -p "$W"; cd "$W"
PY_OMNI=$W/venv_omni/bin/python; PY_VLLM=$W/venv_vllm/bin/python
export HF_HOME=$W/hf
ITEMS=$W/items.jsonl; DESC=$W/desc.jsonl; LAB=$W/labels.jsonl

case "${1:-}" in
setup)
  apt-get update -qq && apt-get install -y -qq ffmpeg tmux git curl python3-venv >/dev/null
  [ -d "$REPO" ] || git clone -q -b main https://github.com/ptrnghieu/icme.git "$REPO"
  git -C "$REPO" pull -q
  [ -d $W/affectgpt ] || git clone -q --depth 1 https://github.com/zeroQiaoba/AffectGPT $W/affectgpt
  [ -d $W/emos ] || git clone -q --depth 1 https://github.com/NLP2CT/EmoS $W/emos
  [ -d $W/multimer ] || git clone -q --depth 1 https://github.com/14H034160212/MultiMER $W/multimer
  python3 -m venv $W/venv_omni && python3 -m venv $W/venv_vllm
  $PY_OMNI -m pip -q install -U pip
  $PY_OMNI -m pip -q install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  $PY_OMNI -m pip -q install -U "transformers==5.24.0" accelerate "qwen-omni-utils[decord]" soundfile librosa audioread av huggingface_hub pandas openpyxl numpy
  $PY_VLLM -m pip -q install -U pip && $PY_VLLM -m pip -q install -U vllm numpy
  echo "== checks"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  $PY_OMNI -c "import torch,transformers; print('torch',torch.__version__,'cuda',torch.version.cuda,'bf16',torch.cuda.is_bf16_supported(),'gpu',torch.cuda.get_device_name(0)); x=torch.randn(64,64,device='cuda',dtype=torch.bfloat16); print('transformers',transformers.__version__)"
  $PY_VLLM -c "import vllm,torch; print('vllm',vllm.__version__,'torch',torch.__version__,'cuda ok',torch.cuda.is_available())"
  df -h $W | tail -1
  ;;
prepare)
  Z26=$($PY_OMNI -c "from huggingface_hub import hf_hub_download; import os; print(hf_hub_download(repo_id='MERChallenge/MER2026', repo_type='dataset', filename='video_7z/video_track2_train_human/video_split.zip', local_dir='$W/MER2026', token=os.environ['HF_TOKEN']))")
  $PY_OMNI $OV/p17_prepare.py --affectgpt $W/affectgpt --emos $W/emos --mer26zip "$Z26" \
     --meld_tar https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz --work $W/media --out $ITEMS
  $PY_OMNI -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-Omni-7B'); s('Qwen/Qwen2.5-7B-Instruct')"
  df -h $W | tail -1
  ;;
timing)
  rm -f $W/timing.jsonl
  for ds in OV-MERD+ MER2025-FG EmoS-MELD; do
    $PY_OMNI $OV/p17_describe.py --items $ITEMS --out $W/timing.jsonl --only_ds $ds --limit 7 ${AUDIO_MODE:+--audio_mode $AUDIO_MODE}
  done
  $PY_OMNI $OV/p17_inspect.py $W/timing.jsonl
  ;;
describe)
  $PY_OMNI $OV/p17_describe.py --items $ITEMS --out $DESC ${AUDIO_MODE:+--audio_mode $AUDIO_MODE}
  $PY_OMNI $OV/p17_inspect.py $DESC
  ;;
describe_seeds)  # optional, later: 2 more sample seeds on OV-MERD+ (greedy and seed 0 are reused)
  $PY_OMNI $OV/p17_describe.py --items $ITEMS --out $DESC --only_ds OV-MERD+ --n_seeds 3 ${AUDIO_MODE:+--audio_mode $AUDIO_MODE}
  ;;
extract)
  CUT=${CUT:-turn}
  $PY_VLLM $OV/p17_extract_vllm.py --descs $DESC --out $LAB --cut $CUT --prompt rl --seed 0
  $PY_VLLM $OV/p17_extract_vllm.py --descs $DESC --out $W/labels_seed1.jsonl --cut $CUT --prompt rl --seed 1
  $PY_VLLM $OV/p17_extract_vllm.py --descs $DESC --out $W/labels_oldprompt.jsonl --cut $CUT --prompt old --seed 0
  ;;
count)  # clips described so far, per dataset
  $PY_OMNI -c "import json,collections; c=collections.Counter(json.loads(l)['ds'] for l in open('$DESC') if json.loads(l).get('ok')); print(dict(c))"
  ;;
extract_ov)  # OV-MERD+ only (run after stopping describe with Ctrl-C once OV-MERD+ is complete; describe resumes later)
  CUT=${CUT:-turn}
  $PY_VLLM $OV/p17_extract_vllm.py --descs $DESC --out $LAB --cut $CUT --prompt rl --seed 0 --only_ds OV-MERD+
  $PY_VLLM $OV/p17_extract_vllm.py --descs $DESC --out $W/labels_seed1.jsonl --cut $CUT --prompt rl --seed 1 --only_ds OV-MERD+
  $PY_VLLM $OV/p17_extract_vllm.py --descs $DESC --out $W/labels_oldprompt.jsonl --cut $CUT --prompt old --seed 0 --only_ds OV-MERD+
  ;;
pack)
  tar -czf $W/p17_out.tar.gz -C $W $(cd $W && ls desc.jsonl labels*.jsonl timing.jsonl 2>/dev/null)
  ls -la $W/p17_out.tar.gz
  ;;
*) echo "usage: bash run_p17.sh setup|prepare|timing|describe|describe_seeds|count|extract_ov|extract|pack"; exit 1 ;;
esac
