# Role
你是一位深度学习专家，擅长用最直观、最底层的 PyTorch 代码解释复杂的 Transformer 架构。

# Task
请为我构建一个完整的、可运行的从零开始的 Transformer 预训练项目。这个项目的核心目标是：通过训练一个字符级的语言模型（Char-GPT），让我彻底理解 Transformer 的每一个数学细节。

# Core Architecture Requirements (必须手动实现，禁止使用 nn.Transformer)
1. **Tokenizer**: 实现一个简单的字符映射表（char-to-idx, idx-to-char），处理纯文本数据。
2. **Embedding**: 实现 Token Embedding 和可学习的 Position Embedding。
3. **Multi-Head Attention (核心)**: 
   - 手动实现 Q、K、V 的线性投影（Linear Projections）。
   - 手动实现缩放点积注意力（Scaled Dot-Product Attention）：$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$。
   - 必须包含下三角掩码（Causal Mask/Triangular Mask），确保模型在预测时看不到未来的信息。
4. **Transformer Block**: 
   - 包含残差连接（Residual Connections）。
   - 包含 Layer Normalization（层归一化）。
   - 包含 Feed-Forward Network（两层线性变换加 GELU 激活）。
5. **Head**: 最后的线性层输出词表大小的 Logits。

# Implementation Details & Debugging
- **Tensor Shape Annotations**: 在 `model.py` 的关键位置（如 Attention 的矩阵乘法处），必须用注释标出张量的维度变化。例如：`# [B, T, C] @ [B, C, T] -> [B, T, T]`。
- **Modular Code**: 按照 `config.py`, `dataset.py`, `model.py`, `train.py`, `generate.py` 分文件编写。
- **Simplicity**: 代码要干净，不要过度封装，尽量让数学公式直接映射到代码行。

# Project Structure
1. `config.py`: 定义超参数（batch_size=32, block_size=64, n_embd=128, n_head=4, n_layer=4）。
2. `dataset.py`: 读取本地 `input.txt`，进行字符编码，返回 (x, y) 训练对。
3. `model.py`: 包含上述所有 Transformer 组件。
4. `train.py`: 实现训练循环，每 500 次迭代输出一次 Loss 并保存模型权重为 `base_model.pth`。
5. `generate.py`: 加载模型，实现简单的自回归生成。

请从 `config.py` 开始，一步步构建整个项目。


### 🎯 进阶指令：从 Pre-training 跨越到 SFT

> **指令：**
> 
> “基于当前项目的 Transformer 架构，我需要补充 **Post-training (SFT)** 阶段的代码。请执行以下任务：
> 
> 1. **Data Schema**: 创建一个 `sft_dataset.py`，支持读取 JSONL 格式的指令对（instruction/answer）。使用模板 `<s>[INST] {msg} [/INST] {ans} </s>` 进行编码。
> 2. **Loss Masking**: 实现 **Target-only Loss** 逻辑。在微调时，仅对 `{ans} </s>` 部分计算 CrossEntropy，忽略指令部分的梯度，以防模型学偏。
> 3. **Trainer**: 编写 `sft_train.py`。要求支持 `load_state_dict` 加载 `base_model.pth`，并使用较小的学习率（如 `5e-5`）进行微调，保存为 `chat_model.pth`。
> 4. **Inference**: 编写 `chat.py` 脚本，实现交互式对话，自动包装 `[INST]` 标签并流式输出结果（如果有）。
> 
> 请确保所有新增代码与现有的 `Config` 和 `CharTokenizer` 完全兼容。”
