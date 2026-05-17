<div align="center">
  <img src="assets/MolmoAct2.svg" alt="MolmoAct2 Logo" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
  <br>
  <br>
  <h1>MolmoAct2: Action Reasoning Models for Real-world Deployment</h1>
</div>

<p align="center">
  <a href="https://github.com/allenai/molmoact2/blob/main/LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/github/license/allenai/molmoact2">
  </a>
  <a href="https://allenai.org/blog/molmoact2">
    <img alt="Blog Post" src="https://img.shields.io/badge/Blog-Post-F0529C">
  </a>
  <a href="https://arxiv.org/abs/2605.02881">
    <img alt="Paper URL" src="https://img.shields.io/badge/arXiv-2605.02881-red?logo=arxiv">
  </a>
  <a href="https://huggingface.co/collections/allenai/molmoact2-models-69f81e05242e2499606b1be6">
    <img alt="Base Models" src="https://img.shields.io/badge/HF-Base%20Models-yellow?logo=huggingface">
  </a>
  <a href="https://huggingface.co/collections/allenai/molmoact2-finetuned-models-69f81e23d5a7b34fde34f2ce">
    <img alt="Finetuned Models" src="https://img.shields.io/badge/HF-Finetuned%20Models-yellow?logo=huggingface">
  </a>
  <a href="https://huggingface.co/collections/allenai/molmoact2-bimanualyam-dataset-69f81e17b140ec34f430a35e">
    <img alt="MolmoAct2-BimanualYAM Dataset" src="https://img.shields.io/badge/HF-MolmoAct2--BimanualYAM%20Dataset-yellow?logo=huggingface">
  </a>
  <a href="https://huggingface.co/collections/allenai/molmoact2-datasets-69f81e316ec3daafe3f9555c">
    <img alt="Robotics Datasets" src="https://img.shields.io/badge/HF-Robotics%20Datasets-yellow?logo=huggingface">
  </a>
  <a href="https://huggingface.co/collections/allenai/molmo2-er-datasets-69f8d605d92d46a5fc24ced2">
    <img alt="ER Datasets" src="https://img.shields.io/badge/HF-ER%20Datasets-yellow?logo=huggingface">
  </a>
</p>

MolmoAct2 is Ai2's open family of action reasoning models for robot control and real-world deployment. It builds on the Molmo2-ER embodied-reasoning vision-language backbone, adds robot state and action modeling, and connects the VLM to a flow-matching continuous action expert for closed-loop manipulation. The release includes base checkpoints for continued training, fine-tuned robot policies for evaluation and deployment, and the datasets used to build MolmoAct2 and Molmo2-ER.

## Models

### Base Models

We provide base checkpoints at every training stage for continued MolmoAct2 training and robot fine-tuning. These are foundation checkpoints rather than one-size-fits-all deployment policies.

| Model | Use Case | Description | Checkpoint Path |
| --- | --- | --- | --- |
| MolmoAct2 | Fine-tuning | Post-trained MolmoAct2 model with a continuous flow-matching action expert. Use as the default foundation checkpoint for adapting to a target robot embodiment or benchmark. | https://huggingface.co/allenai/MolmoAct2 |
| MolmoAct2-Think | Fine-tuning | MolmoAct2 foundation checkpoint with depth-token reasoning. Use when downstream policies should reason over compact depth predictions before acting. | https://huggingface.co/allenai/MolmoAct2-Think |
| MolmoAct2-Pretrain | Post-training | Pre-trained discrete autoregressive VLA backbone before the continuous action expert is attached. Intended for continuing MolmoAct2 training stages, not direct continuous-control inference. | https://huggingface.co/allenai/MolmoAct2-Pretrain |
| Molmo2-ER | Pre-training | Embodied-reasoning VLM backbone used as the starting point for MolmoAct2 action models. | https://huggingface.co/allenai/Molmo2-ER |

### Finetuned Models

We also provide fine-tuned checkpoints for common robot platforms and benchmarks. These models are intended to run directly in their target setting, or to serve as a stronger starting point for closely related robots. As with any robot policy, performance depends on hardware, cameras, calibration, action conventions, and language/task distribution.

| Model | Use Case | Description | Checkpoint Path |
| --- | --- | --- | --- |
| MolmoAct2-DROID | Inference / Fine-tuning | MolmoAct2 fine-tuned on the filtered DROID Franka mixture with absolute joint-pose control. Intended for DROID-style policy inference or further fine-tuning. | https://huggingface.co/allenai/MolmoAct2-DROID |
| MolmoAct2-BimanualYAM | Inference / Fine-tuning | MolmoAct2 fine-tuned on the bimanual YAM mixture with absolute joint-pose control and annotated language instructions. | https://huggingface.co/allenai/MolmoAct2-BimanualYAM |
| MolmoAct2-SO100_101 | Inference / Fine-tuning | MolmoAct2 fine-tuned on SO-100/SO-101 datasets with absolute joint-pose control and annotated language instructions. | https://huggingface.co/allenai/MolmoAct2-SO100_101 |
| MolmoAct2-LIBERO | Inference / Fine-tuning | MolmoAct2 fine-tuned on the full LIBERO training mixture, combining Spatial, Object, Goal, and Long suites. | https://huggingface.co/allenai/MolmoAct2-LIBERO |
| MolmoAct2-Think-LIBERO | Inference / Fine-tuning | MolmoAct2-Think fine-tuned on LIBERO with depth-and-action examples and adaptive depth reasoning. | https://huggingface.co/allenai/MolmoAct2-Think-LIBERO |

## Datasets

| Data | Description | Dataset Path |
| --- | --- | --- |
| MolmoAct2-BimanualYAM Dataset | Collection of bimanual YAM datasets and related resources used for MolmoAct2 bimanual training and evaluation. | https://huggingface.co/collections/allenai/molmoact2-bimanualyam-dataset-69f81e17b140ec34f430a35e |
| MolmoAct2 Robotics Datasets | Robotics datasets for MolmoAct2 training and fine-tuning, including SO-100/SO-101, DROID, MolmoAct Dataset, BC-Z, Bridge, and RT-1. | https://huggingface.co/collections/allenai/molmoact2-datasets-69f81e316ec3daafe3f9555c |
| Molmo2-ER Datasets | Embodied reasoning datasets used for Molmo2-ER and MolmoAct2 backbone training, including spatial, 3D, robotics, and visual reasoning data. | https://huggingface.co/collections/allenai/molmo2-er-datasets-69f8d605d92d46a5fc24ced2 |

Note that all of the robotics datasets for pre-training and post-training are in LeRobot v3.0 format, paired with extra language annotations.

## Evaluation

The MolmoAct2 LeRobot fork is included as a Git submodule at `lerobot/`. After cloning this repository, initialize the submodule from the repo root:

```bash
git submodule update --init --recursive
cd lerobot
```

For LIBERO replication and other evaluation instructions, follow the local LeRobot README at [`lerobot/README.md`](lerobot/README.md).

## Inference server (MolmoAct2-DROID)

This repo ships a FastAPI inference server (`host_server.py`) that loads
[allenai/MolmoAct2-DROID](https://huggingface.co/allenai/MolmoAct2-DROID) and exposes
the same `/act` wire protocol that `inference_script.py` already speaks.

### 1. Install [uv](https://docs.astral.sh/uv/)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL          # reload PATH so the `uv` binary is picked up
uv --version
```

### 2. Create the project environment

The pinned dependencies (CUDA-12.1 PyTorch wheels, `transformers`, `fastapi`,
`json-numpy`, …) live in `pyproject.toml`. From the repo root:

```bash
uv sync                  # creates .venv/ and installs all deps
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expected: True NVIDIA RTX A6000
```

`uv` reads `.python-version` (3.11) and downloads a matching interpreter if needed.
Re-run `uv sync` after pulling new commits.

### 3. Download the checkpoint (~22 GB)

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1                    # fast parallel download
uv run hf download allenai/MolmoAct2-DROID            # -> ~/.cache/huggingface/hub
```

To put the cache on a different disk, set `HF_HOME=/path/to/cache` before the
download (and also when starting the server).

### 4. Start the server

```bash
uv run python host_server.py --host 0.0.0.0 --port 8000 --dtype bfloat16
```

Useful flags:

- `--dtype bfloat16|float16|float32` — default `bfloat16`. The model card uses
  `float32` (~88 GB), which only fits if you have ~96 GB of free VRAM (e.g. two
  empty A6000s with `device_map`). On a single 48 GB A6000, use `bfloat16`.
- `--device cuda:0`
- `--cuda-graph` — enables CUDA-graph capture for the action expert (~2× faster
  per call but needs ~2 GB more VRAM). Disabled by default to coexist with
  other GPU workloads.
- `--no-warmup` — skip the dummy forward pass.

#### bf16 patches

Loading in `bfloat16` is not officially supported by the upstream MolmoAct2
code; `host_server.py` applies two surgical patches to the cached
`modeling_molmoact2.py` at startup:

1. flow-matching trajectory uses the model dtype instead of hardcoded `float32`
   (otherwise the action expert errors with `mat1 and mat2 must have the same dtype`),
2. `_to_array` casts to `float32` before `.numpy()` (numpy has no bf16 dtype).

Both are idempotent and marked with `# patched_bf16_*` comments. They are
re-applied on every server start, so re-downloading the checkpoint won't break
things.

### 5. Reach it from the LAN

Bound to `0.0.0.0`, the server is reachable on every interface of this host.
On this machine the LAN addresses are:

| Interface | Address | Use from |
| --- | --- | --- |
| `enp37s0f1` (wired) | `172.16.0.42:8000` | other hosts on the wired LAN (e.g. the Polymetis NUC) |
| `wlp39s0` (wifi)    | `10.102.245.84:8000` | hosts on the same wifi network |
| `tailscale0`        | `100.90.164.29:8000` | tailnet |

Health check:

```bash
curl http://172.16.0.42:8000/act
# {"status":"ok","repo_id":"allenai/MolmoAct2-DROID","norm_tag":"franka_droid",...}
```

For a real request, use `inference_script.py` — set `MOLMOACT2_URL =
"http://172.16.0.42:8000"` near the top of the file. The wire format
(`json_numpy`-encoded `external_cam`, `wrist_cam`, `instruction`, `state`,
returns `actions` of shape `(N, 8)`) is documented in the docstring of
`host_server.py`.

### Firewall / port

If clients on the LAN can't connect, open the port locally:

```bash
sudo ufw allow from 172.16.0.0/24 to any port 8000 proto tcp   # adjust subnet
```

## Coming Soon

Full code for training, fine-tuning, deployment, evaluation, and more details are coming soon.

## License

This model is licensed under Apache 2.0. It is intended for research and educational use in accordance with Ai2's Responsible Use Guidelines (https://allenai.org/responsible-use).

## Model and Hardware Safety
MolmoAct2 generate robot actions from visual observations and language instructions, but their behavior may vary across embodiments, environments, and hardware configurations. Users should carefully validate model outputs before deployment, especially when operating physical robots or other actuated systems. Where possible, actions should be monitored through interpretable intermediate outputs (adaptive depth map), simulation rollouts, action limits, or other safety checks before execution on hardware. The model’s action space should be bounded by the training data, robot controller limits, and task-specific safety constraints, including limits on speed, workspace, torque, and contact force. Users should follow the hardware manufacturer’s safety guidelines, use appropriate emergency-stop mechanisms, and operate the system only in a safely configured environment with human supervision.

## Contacts

For questions, collaborations, or support, please contact with:
```
{hqfang,duanj1}@cs.washington.edu 
```
Found a bug or have a feature request? Please open a GitHub issue.

## Citation

```bibtex
@misc{fang2026molmoact2actionreasoningmodels,
      title={MolmoAct2: Action Reasoning Models for Real-world Deployment}, 
      author={Haoquan Fang and Jiafei Duan and Donovan Clay and Sam Wang and Shuo Liu and Weikai Huang and Xiang Fan and Wei-Chuan Tsai and Shirui Chen and Yi Ru Wang and Shanli Xing and Jaemin Cho and Jae Sung Park and Ainaz Eftekhar and Peter Sushko and Karen Farley and Angad Wadhwa and Cole Harrison and Winson Han and Ying-Chun Lee and Eli VanderBilt and Rose Hendrix and Suveen Ellawela and Lucas Ngoo and Joyce Chai and Zhongzheng Ren and Ali Farhadi and Dieter Fox and Ranjay Krishna},
      year={2026},
      eprint={2605.02881},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2605.02881}, 
}
```
