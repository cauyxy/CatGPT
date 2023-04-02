## CatGPT：基于中文数据集的InstructGPT复现
<div id="top" align="center">

   | [English](README.md) | [中文](README-zh.md) |
</div>

CatGPT 是一个开源项目，旨在复现 InstructGPT 的 PPO 算法，并在中文数据集上进行训练。项目的名称 "CatGPT" 由 "ChatGPT" 演变而来，去掉 "h"，表示相较于 ChatGPT，本项目的 "Helpful"（有帮助）和 "Harmless"（无害的）特点有所减弱。此外，"CatGPT" 还代表 "Concatenate" 的含义，表示这个项目是由多个项目组合（缝合似乎更合适？）而成的。本项目包括代码、模型和数据的开源。  

CatGPT 为初学者提供了一个了解和体验 InstructGPT 训练流程的绝佳平台。通过现成的数据、代码和模型，这个项目为您深入了解 InstructGPT 世界提供了一个全面且用户友好的入门途径。CatGPT 的易用性使得初学者可以快速上手，探索 InstructGPT PPO 算法的强大功能，并享受一站式的学习体验。借助 CatGPT，您可以迈出在 AI 驱动语言模型领域学习和实践的第一步，轻松掌握 InstructGPT 训练流程的核心概念。

PreTrained Model: [Bloomz-1b1](https://huggingface.co/bigscience/bloomz-1b1)

## 特点
- 完全基于中文数据集的 PPO 训练  
- 代码开源，便于研究和改进
- 模型开源，便于使用和部署
- 数据开源，使用中文语料库

## 待办事项（TODO）

以下是项目的待办事项，我们将继续努力改进和完善 CatGPT：

- 实现 PPO-PTX 算法：由于trlx的限制，当前版本的 CatGPT 仅支持 PPO 算法，我们计划在未来版本中添加对 PPO-PTX 算法原生的支持，以便为用户提供更多选择。
- 使用 LoRA 进行训练：我们计划在未来版本中应用 LoRA 技术，以便更高效地训练 CatGPT 模型。


## Before Training

``` bash
# 克隆仓库
git clone https://github.com/cauyxy/CatGPT.git

# 进入项目目录
cd CatGPT

# 创建虚拟环境
conda env -n catgpt python==3.8

# 激活虚拟环境
conda activate catgpt

# 安装trlx
git clone https://github.com/CarperAI/trlx.git
cd trlx
pip install torch==2.0.0 --extra-index-url https://download.pytorch.org/whl/cu116 # for cuda
pip install -e .
cd ../

# 安装依赖
pip install -r requirements.txt
```

### Training Process

1. Train SFT:
    ```bash
    cd sft/ && deepspeed train_sft.py
    ```
    Checkpoint: [SFT](https://huggingface.co/xinyu66/catgpt_sft)

2. Train Reward Model:
    ```bash
    cd reward_model/ && deepspeed train_rm.py
    ```
    Download reward model checkpoint:
    ```bash
    mkdir reward_model/rm_checkpoint
    wget https://huggingface.co/xinyu66/catgpt-sft/resolve/main/pytorch_model.bin -O reward_model/rm_checkpoint/pytorch_model.bin
    ```__

3. PPO training:
    ```bash
    mkdir ppo
    accelerate launch --config_file configs/default_accelerate_config.yaml trlx_ppo.py
    ```
    Checkpoint: [PPO](https://huggingface.co/xinyu66/catgpt_ppo)

    🩹 Warning: This particular training configuration requires at least 55GB of VRAM and is setup to use 8 GPUs, decrease `batch_size` in case you're running out of memory.

## 结果

下面是使用 CatGPT 生成的一些示例结果。这些图片展示了模型生成的文本在不同输入条件下的表现。

<p align="center">
  <img src="images/sample1.jpeg" alt="Sample 1" width="300" /><br>
  <em>Sample 1: 请教学习方法</em>
</p>

<p align="center">
  <img src="images/sample2.jpeg" alt="Sample 2" width="300" /><br>
  <em>Sample 2: 安慰朋友建议</em>
</p>

<p align="center">
  <img src="images/sample3.jpeg" alt="Sample 3" width="300" /><br>
  <em>Sample 3: 潮流明星问题</em>
</p>

<p align="center">
  <img src="images/sample4.jpeg" alt="Sample 3" width="300" /><br>
  <em>Sample 4: 程序编写问题(需要更改)</em>
</p>

## 致谢

我们要感谢所有对本项目做出贡献的人，包括但不限于提交代码、报告问题和提供想法的人。特别感谢以下项目及其团队，他们的研究和成果为我们提供了宝贵的灵感和技术支持：


- [trlx](https://github.com/CarperAI/trlx)
- [DuReader](https://github.com/baidu/DuReader.git)
- [ColossalAI](https://github.com/hpcaitech/ColossalAI.git)
- [InstructionWild](https://github.com/XueFuzhao/InstructionWild.git)
- [InstructGPT](https://arxiv.org/abs/2203.02155) 

感谢这些优秀的项目，让我们能够在其基础上构建 CatGPT，为中文 NLP 领域贡献力量。

## 加入我们

我们非常欢迎您参与到 CatGPT 项目中来！您的贡献将对本项目产生深远的影响。您可以通过以下方式参与到本项目中：

- 提交代码：优化模型结构、改进算法实现等。
- 提供高质量数据集：为项目提供优质的中文数据集，以提高模型表现。
- 报告问题：在使用过程中遇到的问题和建议，请在 [Issues](https://github.com/cauyxy/CatGPT/issues) 中提交。
- 完善文档：帮助我们改进项目文档，使其更易于理解和使用。

如果您有兴趣为本项目做出贡献，请开启一个PullRequest。我们期待您的加入，一起让 CatGPT 更加强大！
