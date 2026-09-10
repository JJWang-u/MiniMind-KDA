"""
MiniMind 训练数据管线
=====================
为训练四阶段（预训练 → SFT → DPO → GRPO）提供各自的数据集类：

    PretrainDataset   预训练：纯文本自回归，整段序列都计算 loss
    SFTDataset        监督微调：只对 assistant 回复部分计算 loss
    DPODataset        偏好对齐：chosen / rejected 对比样本
    RLAIFDataset      强化学习：只返回 prompt 字符串，由 RL trainer 在线 rollout

所有类都基于 HuggingFace `datasets` 的惰性加载（load_dataset('json')），
不把整个 jsonl 读进内存；tokenize 在 __getitem__ 里按样本即时完成。
"""
from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Sequence, Value

# 禁用 HuggingFace tokenizer 的多进程并行，避免在 DataLoader 多进程环境中产生死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ==========================================================================================
#  pre_processing_chat — 对话前处理：以一定概率随机插入 system 消息
# ==========================================================================================
#  只有当首条消息不是 system 角色时才可能插入，add_system_ratio 控制概率（默认 20%）。
#  引入随机性可提升模型对有/无 system prompt 两种情况的泛化能力；
#  system 内容从预定义的中英文 prompt 池中随机抽取。
#  带 tools（function calling）的数据完整保留，不做处理。
def pre_processing_chat(conversations, add_system_ratio=0.2):
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations):
        return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model.",
    ]
    # 概率性添加 system：首条不是 system 消息时才考虑
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations


# ==========================================================================================
#  post_processing_chat — 对话后处理：清理模板渲染后多余的空 <think> 块
# ==========================================================================================
#  针对带 CoT 格式的模型，apply_chat_template 有时会渲染出 "<think>\n\n</think>\n\n"
#  这样的空思考块占位符。以概率 (1 - empty_think_ratio) = 80% 直接删除该空块，
#  防止模型学到"无意义思考"的坏习惯；保留少量空块让模型也能处理该边界情况。
def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content


# ==========================================================================================
#  PretrainDataset — 自回归预训练数据集
# ==========================================================================================
#  训练目标：Next-Token Prediction（下一个 token 预测）
#  数据格式：{"text": "一段原始文本"}
#  训练特点：
#    - 整段文本的每个位置都参与预测，没有"只学回复"的区分（与 SFT 相反）
#    - 用 BOS/EOS 标记文本边界，让模型学会文本的起止
#    - PAD token 对应的 label 置 -100，CrossEntropyLoss 自动忽略，不产生梯度
#    - labels 直接 clone 自 input_ids（X 和 Y 错位一格：Y[t] = X[t+1]，由模型内部 shift 完成）
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # HuggingFace datasets 的惰性加载，避免一次性读入大文件
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        # Step 1：tokenize 原始文本，留出首尾各 1 个 token 的位置给 BOS/EOS
        tokens = self.tokenizer(
            str(sample['text']),
            add_special_tokens=False,
            max_length=self.max_length - 2,  # 预留 BOS + EOS 的位置
            truncation=True,
        ).input_ids
        # Step 2：拼接 BOS + token 序列 + EOS，构成完整序列
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        # Step 3：右侧用 PAD 补齐到 max_length，保证 batch 内等长
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        # Step 4：labels 与 input_ids 完全相同，但 PAD 位置置 -100（不参与 loss）
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


# ==========================================================================================
#  SFTDataset — 有监督微调数据集
# ==========================================================================================
#  训练目标：让模型学会"只预测 assistant 回复"，忽略 user/system 输入
#  数据格式：{"conversations": [{"role": ..., "content": ..., "reasoning_content"?: ...}]}
#  训练特点：
#    - generate_labels 扫描 bos_id（"<|im_start|>assistant\n"）定位每段回复，
#      仅将 assistant 回复区间（含 EOS）设为有效 label，其余全部 -100。
#      意义：loss 只反映模型对"正确回答"的拟合，用户输入只作为 context。
#    - 支持 function calling：system 消息携带 tools 字段时透传给 apply_chat_template，
#      生成带工具描述的提示词；assistant 的 tool_calls 字段也会被渲染进模板。
#    - 与 PretrainDataset 的关键区别：标签是"稀疏"的，只有 assistant 部分非 -100。
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 显式声明字段 schema：reasoning_content/tools/tool_calls 在大部分样本中缺省，
        # 声明后缺失字段自动填 None，避免 datasets 按第一行样本推断 schema 时不一致
        features = Features({'conversations': [
            {'role': Value('string'), 'content': Value('string'),
             'reasoning_content': Value('string'), 'tools': Value('string'),
             'tool_calls': Value('string')}
        ]})
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        # 预先 tokenize assistant 回复的起始/结束标记，用于 generate_labels 中定位回复区间
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        """用 chat template 把多轮对话渲染成模型输入字符串。

        - 复制原始 conversations，防止修改原数据
        - 提取 system 消息中的 tools（function calling 场景）并透传给模板
        - tool_calls 若以字符串形式存储则 json.loads 还原为对象，供模板渲染
        - add_generation_prompt=False：训练需要完整的 input+output 序列，而非开放续写
        """
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, tools=tools
        )

    def generate_labels(self, input_ids):
        """生成 SFT 训练所需的稀疏标签序列。

        算法逻辑（滑动窗口扫描）：
        1. 初始化全 -100 的 labels，默认所有位置不计算 loss。
        2. 逐位扫描 input_ids，检测是否匹配 bos_id（assistant 回复起始）。
        3. 匹配到 bos_id 后向后扫描，直到找到 eos_id（回复结束）。
        4. 将 [start, end + len(eos_id)) 区间内的 label 设为对应的 input_ids 值，
           即这段 assistant 回复（含 EOS，让模型学会何时停止）参与 loss 计算。
        5. 跳过已处理区间，继续扫描下一段 assistant 回复（支持多轮对话）。
        """
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                # 跳过 bos_id 本身，从 assistant 实际内容开始
                start = i + len(self.bos_id)
                end = start
                # 向后扫描，找到 eos_id 的位置
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 将 assistant 回复（含 EOS）区间的 label 设为真实 token id
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        # Step 1：随机决定是否插入 system prompt（数据增强）
        conversations = pre_processing_chat(sample['conversations'])
        # Step 2：用 chat template 渲染完整对话字符串
        prompt = self.create_chat_prompt(conversations)
        # Step 3：清理可能出现的空 <think> 块
        prompt = post_processing_chat(prompt)
        # Step 4：tokenize 并截断到 max_length，不足则右侧 PAD 补齐
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        # Step 5：生成稀疏标签，只有 assistant 回复部分有有效 label
        labels = self.generate_labels(input_ids)
        # # === 调试打印（每 token 的 X→Y 对，排查模板/label 问题时打开） ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


# ==========================================================================================
#  DPODataset — 直接偏好优化数据集
# ==========================================================================================
#  训练目标：让模型学会"偏好好回答、远离坏回答"，使输出更符合人类偏好
#  数据格式：{"chosen": [{role, content}...], "rejected": [{role, content}...]}
#  训练特点：
#    - 每条样本同时返回 chosen / rejected 两份 tokenized 序列，
#      训练时 DPO loss 最大化两者对数似然之差（以参考模型为基线）
#    - loss_mask 的设计与 SFT 一致：只有 assistant 回复部分为 1，其余为 0，
#      保证对比信号仅来自模型的实际输出部分
#    - 采用"错位"方式构造输入输出对：x 取 [:-1]，y 取 [1:]，即标准自回归格式；
#      mask 同样错位取 [1:]，与 y 对齐
class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # pad_token_id 若不存在则回退到 0，保证补齐操作不会崩溃
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        # 与 SFTDataset 相同：预先 tokenize assistant 回复的起止标记，
        # 用于 generate_loss_mask 中精准定位 assistant 回复区间
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample['chosen']        # 优质回答对话列表：[{role, content}, ...]
        rejected = sample['rejected']    # 劣质回答对话列表，格式同上
        # Step 1：将 chosen / rejected 对话分别渲染为字符串
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        chosen_prompt = post_processing_chat(chosen_prompt)
        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = post_processing_chat(rejected_prompt)
        # Step 2：tokenize 并 padding 到 max_length（统一序列长度，方便 batch）
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        chosen_input_ids = chosen_encoding['input_ids']
        # Step 3：生成 loss mask，只有 assistant 回复部分为 1
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)
        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        # Step 4：构造自回归训练对，x=[:-1] 作为输入，y=[1:] 作为目标；
        #         mask=[1:] 与 y 对齐，决定哪些位置的 loss 计入梯度
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)
        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected,
        }

    def generate_loss_mask(self, input_ids):
        """生成 DPO 训练所需的 loss mask（0/1 二值序列）。

        与 SFTDataset.generate_labels 逻辑完全相同，区别在于：
        - SFT 返回具体 token id（用于 CE loss）
        - DPO 返回 0/1 掩码（用于 masked 对数似然计算）
        算法：扫描 bos_id → 找到 eos_id → 区间内置 1（含 EOS），其余置 0。
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 将 assistant 回复（含 EOS）区间的 mask 置 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


# ==========================================================================================
#  RLAIFDataset — 强化学习数据集（GRPO 用）
# ==========================================================================================
#  训练目标：为 RL 训练提供 prompt，由 actor 在线采样生成回复，再由 reward model 打分
#  数据格式：{"conversations": [{"role": ..., "content": ...}, ...]}
#  训练特点（与前三个 Dataset 的核心区别）：
#    - **不做离线 tokenize**：只返回 prompt 字符串，由 RL trainer 在线 rollout 时
#      自行 tokenize——RL 需要动态生成回复并实时打分，无法预先固定 token 序列。
#    - create_chat_prompt 剥离最后一条 assistant 消息，将剩余对话渲染为
#      add_generation_prompt=True 的 prompt 供 actor 续写。
#    - thinking_ratio：按概率开启 thinking（模板渲染 <think>\n 引导模型思考，
#      与训练 tokenizer 的 chat_template 里 open_thinking 参数联动）。
class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio  # 按概率开启 thinking
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        """从对话列表中构造 prompt（剥离最后一条 assistant 回复作为续写目标）。

        - 先走 pre_processing_chat（随机插入 system）
        - 随机决定本轮是否开启 thinking（open_thinking 模板参数）
        - conversations[:-1]：去掉最后一条 assistant 消息，只保留上下文
        - add_generation_prompt=True：在末尾追加 "<|im_start|>assistant\n" 续写引导，
          告诉模型"现在开始生成"
        """
        conversations = pre_processing_chat(conversations)
        use_thinking = random.random() < self.thinking_ratio
        return self.tokenizer.apply_chat_template(
            conversations[:-1],
            tokenize=False,
            open_thinking=use_thinking,
            add_generation_prompt=True
        )

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])
        # 返回原始字符串，不做 tokenize，由 RL trainer 在线处理
        return {'prompt': prompt, 'answer': ""}


if __name__ == "__main__":
    pass
