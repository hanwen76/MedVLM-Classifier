# MedVLM-Classifier

一个面向医学图像分类的 VLM 微调框架，核心设计遵循论文《Why are Visually-Grounded Language Models Bad at Image Classification?》的启发：

- 训练目标使用生成式 next-token prediction loss
- 默认冻结 Vision Encoder 与 LLM，仅微调 Projector
- 使用混合采样（分类数据 + instruction-tuning 数据）缓解灾难性遗忘
- 提供闭域分类评估（评估 prompt 不包含 GT 标签，基于模型新生成文本判分）
- 预留 ImageWikiQA 风格检索接口（分类后拉取本地临床指南）

## 1. 项目结构

```text
MedVLM-Classifier/
├── train.py
├── eval.py
├── src/medvlm_classifier/
│   ├── data/               # loader / 格式转换 / mixed sampler / collator
│   ├── model/              # 模型加载 + 冻结策略
│   ├── training/           # 训练入口
│   ├── eval/               # 闭域评估
│   ├── agents/             # ImageWikiQA 检索扩展
│   └── tools.py            # 图像-标签对 -> 指令格式转换工具
├── configs/
├── scripts/
├── examples/
└── knowledge_base/
```

## 2. 安装

```bash
cd MedVLM-Classifier
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. 数据格式

### 分类数据（CSV/JSON/JSONL）

最简 JSON 示例：

```json
[
  {"image": "demo_eczema.jpg", "label": "湿疹"},
  {"image": "demo_psoriasis.jpg", "label": "银屑病"}
]
```

### 自动转换为文本对齐格式

运行：

```bash
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
python -m medvlm_classifier.tools \
  --input_json examples/medical_cls_train.json \
  --output_jsonl outputs/medical_cls_instruction.jsonl
```

生成文本形态：

```text
USER: <image>
这张医学影像显示了什么？
ASSISTANT: <疾病名称>
```

### 指令数据（JSONL）

```json
{"image": "demo_eczema.jpg", "text": "USER: <image>\\n请描述图像。\\nASSISTANT: ..."}
```

### MedMNIST 下载与预处理

导出为本项目可直接读取的格式（`images/` + `train/val/test.json`）：

```bash
python scripts/prepare_medmnist.py \
  --dataset pathmnist \
  --size 224 \
  --output_root data/medmnist \
  --save_rgb
```

输出示例：

- `data/medmnist/pathmnist/images/*.png`
- `data/medmnist/pathmnist/train.json`
- `data/medmnist/pathmnist/val.json`
- `data/medmnist/pathmnist/test.json`

一次导出多个 MedMNIST 子集（示例）：

```bash
python scripts/prepare_medmnist.py --dataset pathmnist --size 224 --output_root data/medmnist --save_rgb
python scripts/prepare_medmnist.py --dataset dermamnist --size 224 --output_root data/medmnist --save_rgb
python scripts/prepare_medmnist.py --dataset bloodmnist --size 224 --output_root data/medmnist --save_rgb
```

### LLaVA-Med 下载后转换

先按官方仓库下载数据（`llava_med_instruct_10k/60k.json` 与图片），再转换为本项目 `instruction_jsonl`：

```bash
python scripts/prepare_llava_med.py \
  --input_json /path/to/llava_med_instruct_10k.json \
  --image_root /path/to/llava_med_images \
  --output_jsonl data/llava_med/instruction_10k_converted.jsonl \
  --require_image_exists \
  --first_round_only
```

如果你下载的是 Hugging Face 分片（如 `train-00000-of-00014.parquet`），`--input_json` 可以直接给该目录或单个 `.parquet` 文件。

对于 `messages + images`（图片嵌入在 parquet 中）的数据，脚本会自动导出图片并生成 jsonl：

```bash
python scripts/prepare_llava_med.py \
  --input_json /path/to/llava-med-zh-instruct-60k/data \
  --output_jsonl data/llava_med/instruction_zh_60k.jsonl \
  --embedded_image_dir data/llava_med/images \
  --first_round_only
```

### 零微调基线评测（MedMNIST）

不微调模型，直接测试 VLM 的 MedMNIST 分类性能：

```bash
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
bash scripts/benchmark_medmnist_zeroshot.sh
```

或一次评测多个模型：

```bash
python scripts/benchmark_medmnist_zeroshot.py \
  --model_ids "llava-hf/llava-1.5-7b-hf,/home/zhanghanwen/models/Qwen2-VL-2B-Instruct" \
  --data_json "data/medmnist/pathmnist/test.json" \
  --image_root "data/medmnist/pathmnist/images" \
  --labels_json "data/medmnist/pathmnist/labels.json" \
  --mode choice \
  --language en \
  --batch_size 1 \
  --output_dir "outputs/medmnist_zeroshot"
```

输出包含：

- `outputs/medmnist_zeroshot/*predictions.jsonl`
- `outputs/medmnist_zeroshot/*summary.json`
- `outputs/medmnist_zeroshot/leaderboard.json`

### CLIP 零微调分类基线

使用 CLIP 图文相似度做闭域 zero-shot 分类：

```bash
python scripts/benchmark_clip_zeroshot.py \
  --clip_model openai/clip-vit-large-patch14 \
  --data_json data/isic2019/test.json \
  --image_root data/isic2019/raw \
  --labels_json data/isic2019/labels.json \
  --prompt_template "a dermoscopic image of {label}" \
  --batch_size 32 \
  --output_dir outputs/isic2019_clip_zeroshot
```

输出包含：

- `outputs/isic2019_clip_zeroshot/predictions.jsonl`
- `outputs/isic2019_clip_zeroshot/summary.json`

`summary.json` 会报告 `exact_match_accuracy`、`macro_f1`、`ambiguity_rate` 等严格分类指标。

### ISIC 2019 下载与预处理

官方数据页：

- [ISIC Challenge 2019](https://challenge.isic-archive.com/landing/2019/)
- [ISIC Challenge Data](https://challenge.isic-archive.com/data/)

可直接下载官方文件：

```bash
mkdir -p data/isic2019/downloads
cd data/isic2019/downloads

wget -O ISIC_2019_Training_Input.zip \
  https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_Input.zip
wget -O ISIC_2019_Training_GroundTruth.csv \
  https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_GroundTruth.csv
wget -O ISIC_2019_Test_Input.zip \
  https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Test_Input.zip
wget -O ISIC_2019_Test_GroundTruth.csv \
  https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Test_GroundTruth.csv
```

转换为本项目可直接读取的格式：

```bash
python scripts/prepare_isic2019.py \
  --train_input data/isic2019/downloads/ISIC_2019_Training_Input.zip \
  --train_gt data/isic2019/downloads/ISIC_2019_Training_GroundTruth.csv \
  --test_input data/isic2019/downloads/ISIC_2019_Test_Input.zip \
  --test_gt data/isic2019/downloads/ISIC_2019_Test_GroundTruth.csv \
  --output_root data/isic2019 \
  --label_style full
```

输出结构：

- `data/isic2019/train.json`
- `data/isic2019/val.json`
- `data/isic2019/test.json`
- `data/isic2019/labels.json`
- `data/isic2019/raw/`

说明：

- 默认会从官方训练集里按类别分层切出 `val.json`
- 默认会丢弃官方测试集中的 `UNK` 类，以匹配当前项目的闭域分类设定
- 如果你要保留 `UNK` 做开放集分析，可加 `--keep_unknown_test`

## 4. 训练（默认只训练 Projector）

```bash
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
bash scripts/train_projector_only.sh
```

或直接：

```bash
python train.py \
  --model_name_or_path llava-hf/llava-1.5-7b-hf \
  --classification_ann examples/medical_cls_train.json \
  --classification_image_root examples/images \
  --instruction_jsonl examples/instruction_tune.jsonl \
  --instruction_image_root examples/images \
  --ratio 2:1
```

默认关键策略在 `freeze_all_except_projector(...)`：

- Vision Encoder `requires_grad=False`
- LLM `requires_grad=False`
- Projector 参数 `requires_grad=True`

若 projector 命名不匹配，可通过 `--projector_keywords` 补充关键词。

### 多 MedMNIST 子集一键训练+评估

```bash
bash scripts/run_medmnist_multi_dataset.sh \
  --model_name_or_path /home/zhanghanwen/models/llava-1.5-7b-hf \
  --instruction_jsonl /home/zhanghanwen/datasets/llava-med-zh-instruct-60k/instruction_zh_60k.jsonl \
  --instruction_image_root /home/zhanghanwen/datasets/llava-med-zh-instruct-60k/images \
  --datasets pathmnist,dermamnist,bloodmnist \
  --medmnist_root data/medmnist \
  --output_root outputs/medmnist_multiset
```

每个数据集都会写入独立目录：

- `outputs/medmnist_multiset/<dataset>/checkpoint/`
- `outputs/medmnist_multiset/<dataset>/eval_closed_world.json`

### ISIC 2019 训练+评估

```bash
bash scripts/run_isic2019.sh \
  --model_name_or_path /path/to/llava-or-qwen-vl \
  --instruction_jsonl /path/to/instruction.jsonl \
  --instruction_image_root /path/to/instruction_images \
  --isic_root data/isic2019 \
  --output_dir outputs/isic2019
```

默认读取：

- `data/isic2019/train.json`
- `data/isic2019/test.json`
- `data/isic2019/raw/`

## 5. 闭域分类评估

```bash
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
bash scripts/eval_closed_world.sh
```

评估时会使用独立的推理数据集构造 prompt：

```text
USER: <image>
这张医学影像显示了什么？
ASSISTANT:
```

不会把 ground-truth 标签写入输入文本，判分基于模型新生成的回答而不是整段输入回显。

## 6. ImageWikiQA 风格扩展

`src/medvlm_classifier/agents/imagewikiqa.py` 提供本地知识库检索器：

- 输入：模型分类疾病名
- 输出：对应临床指南与参考信息

`src/medvlm_classifier/agents/pipeline.py` 提供组合式接口：

1. 先分类
2. 再检索本地知识库
3. 返回结构化结果

## 7. 参考仓库对齐接口（VLMClassifier-main 风格）

为了方便你参考 `VLMClassifier-main`，新增了兼容入口：

- 推理脚本：`python -m medvlm_classifier.eval.reference_infer`
- 示例命令：`bash scripts/reference_infer.sh`
- 输出评估：`bash scripts/reference_eval.sh`

输入格式与参考仓库一致（JSONL）：

```json
{"image": "examples/images/demo_eczema.jpg", "label": "湿疹", "split": "valid"}
```

类别列表为 JSON 数组：

```json
["湿疹", "银屑病"]
```

该兼容入口支持：

- `--including_label` / `--n_labels` / `--fixed_order`
- `--chain_of_thought`
- `--resume`（断点续跑）
- 不同模型 prompt 模板切换（LLaVA/BLIP/InstructBLIP）

## 8. 注意事项

- 示例中的 `examples/images/*.jpg` 仅作占位，请替换为真实图像。
- 大模型训练建议启用 GPU，并按显存调整 `batch_size/max_length`。
- 本项目是框架骨架，可继续接入 LoRA、DeepSpeed、分布式训练等能力。
# MedVLM-Classifier
