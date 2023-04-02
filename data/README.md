# Data 文件夹

本文件夹包含了用于训练 CatGPT 模型三个阶段的数据集示例。

## 文件说明

- `instinwild_ch_example.json`: 包含了 InstructGPT 格式的中文数据集示例，用于训练和评估 CatGPT 模型。
- `only_prompts_example.json`: 包含了仅有提示信息的中文数据集示例，可用于生成文本。
- `train_pairs_example.json`: 包含了成对的中文训练数据示例，用于训练 CatGPT 模型。

## 数据格式

### instinwild_ch_example.json

该文件包含了符合 InstructGPT 格式的中文数据，每个数据包括以下字段：

- `id`: 数据的唯一标识符。
- `instruct`: 指令文本，包含了问题或任务的描述。
- `input`: 本数据集中为空
- `output`: 目标文本，即模型期望生成的正确答案或完成任务的文本。

### only_prompts_example.json

该文件包含了仅有提示信息的中文数据，每个数据包括以下字段：

- `instruct`: 指令文本，包含了问题或任务的描述。

### train_pairs_example.json

该文件包含了成对的中文训练数据，每个数据包括以下字段：

- `prompt`: 输入文本，包含了问题或任务的描述。
- `chosen`: 选择文本，即相对于rejected期望模型生成的正确答案或完成任务的文本。
- `reject`: 拒绝文本，即相对于chosen不期望模型生成的正确答案或完成任务的文本。

## 数据准备

为了训练您自己的 CatGPT 模型，您需要从[HuggingFace Dataset](https://huggingface.co/datasets/xinyu66/catgpt)下载对应的数据,或者通过`process_dureader.py`处理原始数据后，放在本文件夹中，以供训练使用。

## 注意事项

在使用本项目进行训练时，请注意 `train_pairs.json` 数据集来源于百度知道，数据质量堪忧。本项目主要关注于复现 PPO 算法并使其在中文数据集上运行，因此，在数据质量方面可能存在一定的不足。在使用该数据集进行训练时，请您留意可能出现的效果不佳的情况。您可以尝试使用其他高质量的中文数据集进行训练，以获得更好的效果，也欢迎您提供更好的数据集。

## Reference

- [DuReader](https://github.com/baidu/DuReader.git)
- [InstructionWild](https://github.com/XueFuzhao/InstructionWild.git)
