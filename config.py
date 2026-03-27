"""
超参数与训练配置（Char-GPT 字符级语言模型）。
"""

from dataclasses import asdict, dataclass, fields


@dataclass
class Config:
    # 数据与批次
    batch_size: int = 32
    block_size: int = 64  # 上下文长度（时间步数 T）

    # 模型
    n_embd: int = 128  # 嵌入维度 C
    n_head: int = 4
    n_layer: int = 4
    dropout: float = 0.0  # 教学用可设为 0，便于对照公式

    # 优化
    learning_rate: float = 3e-4
    max_iters: int = 5000
    weight_decay: float = 0.01

    # 日志与保存
    log_interval: int = 500
    device: str = "cuda"  # 无 GPU 时在 train.py 中回退到 cpu

    # 数据文件
    data_path: str = "input.txt"
    checkpoint_path: str = "base_model.pth"

    # SFT（指令微调，见 sft_train.py）
    sft_data_path: str = "sft_data.jsonl"
    sft_learning_rate: float = 5e-5
    sft_max_iters: int = 2000
    sft_log_interval: int = 100
    sft_weight_decay: float = 0.01
    chat_checkpoint_path: str = "chat_model.pth"
    base_checkpoint_path: str = "base_model.pth"  # SFT 初始化权重

    # 交叉熵忽略标签（仅对 answer 计 loss 时使用）
    ignore_index: int = -100


def merge_config(saved: dict | None) -> Config:
    """用当前 Config 默认值补全旧 checkpoint 中缺失字段。"""
    base = asdict(Config())
    names = {f.name for f in fields(Config)}
    if saved:
        for k, v in saved.items():
            if k in names:
                base[k] = v
    return Config(**base)
