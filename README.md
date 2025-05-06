# MultiDiffusion

一个跨框架实现的改进型扩散概率模型(DDPM)，支持多种数据集和灵活的配置选项。

## 项目特点

- 🚀 支持 PyTorch 和 PaddlePaddle 双框架实现
- 📊 支持 MNIST、CIFAR-10/100、CelebA、LSUN 等多个数据集
- ⚙️ 灵活的噪声调度策略（Linear、Cosine、自定义）
- 🎯 支持单机/多机多卡分布式训练
- 📈 内置完整的评估指标（FID、LPIPS）
- 🎨 丰富的可视化工具

## 安装要求

- Python 3.12+
- CUDA 支持（推荐）
- 依赖包：见 requirements.txt

## 快速开始

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/MultiDiffusion.git
cd MultiDiffusion
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 训练模型：
```bash
python train.py --config configs/mnist.yaml
```

4. 生成图像：
```bash
python generate.py --checkpoint path/to/checkpoint --samples 16
```

## 项目结构

```
MultiDiffusion/
├── configs/                # 配置文件目录
├── multidiffusion/        # 核心代码
│   ├── models/            # 模型定义
│   ├── data/             # 数据加载和处理
│   ├── utils/            # 工具函数
│   └── trainers/         # 训练器
├── scripts/               # 训练和评估脚本
├── tests/                # 单元测试
└── notebooks/            # 示例笔记本
```

## 配置说明

在 `configs/` 目录下提供了多个预设配置文件：

- `mnist.yaml`: MNIST 数据集配置
- `cifar10.yaml`: CIFAR-10 数据集配置
- `celeba.yaml`: CelebA 数据集配置

可以通过修改配置文件自定义：
- 数据集参数
- 模型架构
- 训练超参数
- 噪声调度策略

## 评估指标

本项目提供以下评估指标：

- FID (Fréchet Inception Distance)
- LPIPS (Learned Perceptual Image Patch Similarity)
- 生成样本多样性分析

## 可视化工具

1. TensorBoard 监控：
```bash
tensorboard --logdir runs/
```

2. Gradio 演示：
```bash
python app.py
```

## 开发路线图

- [x] 基础 DDPM 实现
- [x] 多种噪声调度策略
- [x] 分布式训练支持
- [ ] Latent Diffusion 支持
- [ ] Classifier-Free Guidance
- [ ] 在线演示部署

## 贡献指南

欢迎提交 Pull Request 或 Issue！

## 许可证

MIT License 