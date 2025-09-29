import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import copy
import math
from tqdm import tqdm
from typing import List, Tuple, Union
from func_3d.utils import ORGAN_GROUPS
from abc import ABC, abstractmethod


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=4):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA weights
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=4, alpha=4):
        super().__init__()
        self.base_layer = copy.deepcopy(linear_layer)
        self.lora = LoRALayer(
            linear_layer.in_features,
            linear_layer.out_features,
            rank=rank,
            alpha=alpha
        )

    def forward(self, x):
        return self.base_layer(x) + self.lora(x)


class LoRALinear_Nores(nn.Module):
    def __init__(self, linear_layer, rank=4, alpha=4):
        super().__init__()
        self.lora = LoRALayer(
            linear_layer.in_features,
            linear_layer.out_features,
            rank=rank,
            alpha=alpha
        )

    def forward(self, x):
        return self.lora(x)


class ExpertBlock(nn.Module):
    def __init__(self, base_block, rank=4, add_baseres=True):
        super().__init__()
        self.base_block = copy.deepcopy(base_block)
        self.add_baseres = add_baseres
        # 为block中的线性层添加LoRA
        self._add_lora_layers(self.base_block, rank)

    def _add_lora_layers(self, module, rank, ):
        for name, child in module.named_children():
            if name == 'qkv':
                if self.add_baseres:
                    setattr(module, name, LoRALinear(child, rank=rank))
                else:
                    setattr(module, name, LoRALinear_Nores(child, rank=rank))
            else:
                self._add_lora_layers(child, rank)

    def forward(self, x):
        return self.base_block(x)

# 两层门控
class GatingNetwork_twolevel(nn.Module):
    def __init__(self, seq_len, hidden_size, num_experts, group_num, text_embed_dim=768, use_text=False):
        super().__init__()
        self.num_experts = num_experts
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.text_embed_dim = text_embed_dim
        self.text_projection = nn.Linear(text_embed_dim, hidden_size)
        self.image_projection = nn.Linear(seq_len * hidden_size, hidden_size)
        self.fuse = nn.Linear(num_experts * hidden_size, hidden_size)

        if use_text:
            # self.gating_weights = nn.Parameter(
            #     torch.randn((num_experts * seq_len * hidden_size) + hidden_size, num_experts) /
            #     math.sqrt((num_experts * seq_len * hidden_size) + hidden_size)
            # )
            # 第一层门控 - 用于粗分类

            self.num_first_level = group_num
            self.num_second_level = num_experts // group_num
            input_dim = hidden_size * 2
            self.top_gate = nn.Linear(input_dim, self.num_first_level)
            # 第二层门控 - 每个粗类别下的细分类
            self.sub_gates = nn.ModuleList([
                nn.Linear(input_dim, self.num_second_level)
                for _ in range(group_num)
            ])

        else:
            self.gating_weights = nn.Parameter(
                torch.randn((num_experts * seq_len * hidden_size), num_experts) /
                math.sqrt((num_experts * seq_len * hidden_size))
            )

        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def get_hierarchical_weights(self, x):
        batch_size = x.shape[0]

        # 计算顶层分组概率
        top_logits = self.top_gate(x)
        top_probs = F.softmax(top_logits, dim=-1)  # [batch_size, num_first_level]

        # 计算每个分组内的细分概率
        sub_logits = torch.stack([gate(x) for gate in self.sub_gates], dim=1)
        sub_probs = F.softmax(sub_logits, dim=-1)  # [batch_size, num_first_level, num_second_level]

        # 计算最终的扁平化专家权重
        final_probs = (top_probs.unsqueeze(-1) * sub_probs).view(batch_size, -1)  # [batch_size, expert_num]

        return final_probs, top_probs, sub_probs, top_logits

    def compute_hierarchical_loss(self, top_logits, sub_probs, top_labels):
        """
        计算两部分损失：
        1. 顶层router的分类交叉熵损失
        2. 每个组内的负载均衡损失

        Args:
            top_logits: 顶层router的输出logits [batch_size, num_first_level]
            sub_probs: 每个组内的概率分布 [batch_size, num_first_level, num_second_level]
            top_labels: 顶层分类的标签 [batch_size]
        """
        # 1. 顶层分类损失
        classification_loss = F.cross_entropy(top_logits, top_labels)

        # 2. 组内负载均衡损失
        balance_loss = 0
        for group in range(self.num_first_level):
            # 计算每个组内专家的平均使用率
            group_probs = sub_probs[:, group, :]  # [batch_size, num_second_level]
            mean_usage = group_probs.mean(0)  # [num_second_level]

            # 计算该组的负载均衡损失
            # 使用KL散度使专家使用率趋向均匀分布
            target = torch.ones_like(mean_usage) / self.num_second_level
            group_balance_loss = F.kl_div(
                mean_usage.log(),
                target,
                reduction='sum'
            )
            balance_loss += group_balance_loss

        # 组合损失（可以添加权重系数）
        total_loss = classification_loss + 0.01 * balance_loss

        return total_loss

    def forward(self, expert_outputs, text_embedding=None, top_labels=None):
        batch_size = expert_outputs[0].shape[0]
        HW = expert_outputs[0].shape[1]

        expert_stacked = torch.stack(expert_outputs, dim=-1)  # B,h*w,c,8
        del expert_outputs
        stacked = expert_stacked.reshape(batch_size, HW, -1)
        stacked = self.fuse(stacked)  # B,h*w,c
        L, d = self.seq_len, self.hidden_size
        features = stacked.reshape(batch_size, L * d)
        features = self.image_projection(features)
        del stacked
        if text_embedding is not None:
            projected_text = self.text_projection(text_embedding)
            features = torch.cat([features, projected_text], dim=1)
            del projected_text

        norm = torch.norm(features, p=2, dim=1, keepdim=True)
        E_omega = features / norm.clamp(min=1e-6)
        del features, norm

        final_probs, top_probs, sub_probs, top_logits = self.get_hierarchical_weights(E_omega)
        del E_omega
        # 计算损失
        loss = self.compute_hierarchical_loss(top_logits, sub_probs, top_labels)

        output = torch.einsum('bshx,bx->bsh', expert_stacked, final_probs)
        # epsilon = E_omega @ self.gating_weights
        # gates = F.softmax(epsilon / self.temperature.clone(), dim=-1)
        return output, loss


class BaseGatingNetwork(nn.Module, ABC):
    def __init__(self, seq_len, hidden_size, num_experts, text_embed_dim=768, use_text=False):
        super().__init__()
        self.num_experts = num_experts
        self.experts_per_group = 2
        self.num_groups = num_experts // self.experts_per_group
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        # 共同的网络层
        self.text_projection = nn.Linear(text_embed_dim, hidden_size)
        self.image_projection = nn.Linear(seq_len * hidden_size, hidden_size)
        self.fuse = nn.Linear(num_experts * hidden_size, hidden_size)

        # gate网络
        input_dim = hidden_size * 2 if use_text else hidden_size
        self.gate = nn.Linear(input_dim, num_experts)

    def compute_gating_loss(self, logits, masked_logits, group_ids):
        """共同的loss计算逻辑"""
        # 1. 分类loss
        target_mask = torch.zeros_like(logits)
        start_idx = group_ids[0] * self.experts_per_group
        end_idx = start_idx + self.experts_per_group
        target_mask[:, start_idx:end_idx] = 1.0

        classification_loss = F.binary_cross_entropy_with_logits(
            logits,
            target_mask,
            reduction='mean'
        )

        # 2. 负载均衡loss
        scores = F.softmax(masked_logits, dim=1)
        group_scores = scores[:, start_idx:end_idx]
        mean_usage = group_scores.mean(0)
        target = torch.ones_like(mean_usage) / self.experts_per_group
        balance_loss = F.kl_div(
            mean_usage.log(),
            target,
            reduction='sum'
        )

        total_loss = classification_loss + 0.01 * balance_loss
        return total_loss, classification_loss, balance_loss

    def process_features(self, features, text_embedding=None):
        """共同的特征处理逻辑"""
        batch_size = features.shape[0]
        features = features.reshape(batch_size, self.seq_len * self.hidden_size)
        features = self.image_projection(features)

        if text_embedding is not None:
            projected_text = self.text_projection(text_embedding)
            features = torch.cat([features, projected_text], dim=1)

        norm = torch.norm(features, p=2, dim=1, keepdim=True)
        E_omega = features / norm.clamp(min=1e-6)
        return E_omega

    def compute_gates(self, E_omega, group_ids):
        """共同的gate计算逻辑"""
        logits = self.gate(E_omega)

        gate_mask = torch.zeros_like(logits)
        for i, gid in enumerate(group_ids):
            start_idx = gid * self.experts_per_group
            gate_mask[i, start_idx:start_idx + self.experts_per_group] = 1.0

        masked_logits = logits.masked_fill(gate_mask == 0, float('-inf'))
        gates = F.softmax(masked_logits, dim=-1)

        loss, cls_loss, balance_loss = self.compute_gating_loss(logits, masked_logits, group_ids)
        return gates, (loss, cls_loss, balance_loss), logits, masked_logits

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass


class GatingNetwork(BaseGatingNetwork):
    def forward(self, expert_outputs, text_embedding=None, group_ids=None):
        batch_size = expert_outputs[0].shape[0]
        HW = expert_outputs[0].shape[1]

        # 处理expert输出
        expert_stacked = torch.stack(expert_outputs, dim=-1)
        stacked = expert_stacked.reshape(batch_size, HW, -1)
        stacked = self.fuse(stacked)

        # 特征处理
        E_omega = self.process_features(stacked, text_embedding)

        # 计算gates
        gates, losses, _, _ = self.compute_gates(E_omega, group_ids)

        # 计算输出
        output = torch.einsum('bshx,bx->bsh', expert_stacked, gates)

        return output, losses


class GatingNetwork_pre(BaseGatingNetwork):
    def forward(self, x, text_embedding=None, group_ids=None):
        # 特征处理
        E_omega = self.process_features(x, text_embedding)

        # 计算gates
        gates, losses, _, _ = self.compute_gates(E_omega, group_ids)

        return gates, losses


class MoEBlock(nn.Module):
    def __init__(self, base_block, seq_len, hidden_size, num_experts=3, lora_rank=4, use_text=False, pre_gating=True):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.pre_gating = pre_gating

        self.base_block = base_block
        self.experts = nn.ModuleList([
            ExpertBlock(base_block, rank=lora_rank, add_baseres=False)
            for _ in range(num_experts)
        ])

        self.gating = GatingNetwork(
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_experts=num_experts,
            text_embed_dim=768,
            use_text=use_text,
        )

        self.gating_pre = GatingNetwork_pre(
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_experts=num_experts,
            text_embed_dim=768,
            use_text=use_text,
        )

    def forward(self, x, text_embedding=None, top_labels=0):
        base_output = self.base_block(x)
        if isinstance(base_output, tuple):
            base_output = base_output[0]

        H, W = base_output.shape[1], base_output.shape[2]
        base_output = base_output.reshape(base_output.shape[0], -1, base_output.shape[-1])
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(x)
            expert_out = expert_out.reshape(expert_out.shape[0], -1, expert_out.shape[-1])
            expert_outputs.append(expert_out)
        if self.pre_gating:
            gates, loss = self.gating_pre(x, text_embedding, top_labels)
            stacked_outputs = torch.stack(expert_outputs, dim=-1)
            gated_output = torch.einsum('bshx,bx->bsh', stacked_outputs, gates)
        else:
            gated_output, loss = self.gating(expert_outputs, text_embedding, top_labels)

        # 清理中间变量
        del expert_outputs

        output = base_output + gated_output  # bsh
        output = output.reshape(-1, H, W, output.shape[-1])
        # 清理更多中间变量
        del base_output, gated_output
        torch.cuda.empty_cache()
        return output, loss # loss,tuple (loss, cls_loss, balance_loss)


class TextEncoder:
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, text):
        # 如果输入是单个字符串，转换为列表
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            if isinstance(text, str):
                text = [text]

            # Tokenize
            tokens = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            # 获取BERT输出
            outputs = self.model(**tokens)

            # 使用[CLS]令牌的最后隐藏状态作为文本嵌入
            text_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
            # 清理中间变量
            del outputs, tokens
            torch.cuda.empty_cache()

            return text_embeddings


class MoELoRAHieraDet(nn.Module):
    def __init__(self, base_encoder, img_size=512, num_experts=3, lora_rank=4, use_text=False,
                 pretrained_model='./checkpoints/sam2_hiera_tiny.pt'):
        super().__init__()
        self.load_base_encoder_weights(base_encoder, pretrain_path=pretrained_model)
        self.trunk = base_encoder.trunk
        self.neck = base_encoder.neck
        self.scalp = base_encoder.scalp
        self.stages = self.trunk.stages
        self.dims = self.trunk.channel_list[::-1]
        self.num_stages = len(self.stages)
        self.stage_ends = [sum(self.stages[:i]) - 1 for i in range(1, len(self.stages) + 1)]
        self.use_text = use_text

        self.text_encoder = TextEncoder()

        self.blocks = nn.ModuleList()
        curr_dim_idx = 0
        curr_resolution = img_size // 4

        for stage_idx in range(self.num_stages):
            seq_len = curr_resolution * curr_resolution

            for block_idx in range(self.stages[stage_idx]):
                # 确定是否使用MoE的策略
                use_moe = self._should_use_moe(stage_idx, block_idx, self.stages[stage_idx])

                if use_moe:
                    # 选择gating的输入，是x还是每个experts的总输出
                    pre_gating = False
                    moe_block = MoEBlock(
                        base_block=self.trunk.blocks[curr_dim_idx + block_idx],
                        seq_len=seq_len if not pre_gating else seq_len * 4,
                        hidden_size=self.dims[stage_idx] if not pre_gating else self.dims[stage_idx-1],
                        num_experts=num_experts,
                        lora_rank=lora_rank,
                        use_text=use_text,
                        pre_gating=pre_gating
                    )
                    self.blocks.append(moe_block)
                else:
                    # 对于不使用MoE的层，直接使用原始block
                    self.blocks.append(self.trunk.blocks[curr_dim_idx + block_idx])
            curr_dim_idx += self.stages[stage_idx]
            curr_resolution = curr_resolution // 2

    def load_base_encoder_weights(self, base_encoder, pretrain_path):
        """
        加载预训练权重到base_encoder

        Args:
            base_encoder: 基础编码器模型
            pretrain_path: 预训练权重文件路径
        """
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        state_dict = checkpoint['model']
        # 只保留包含image_encoder的权重，并处理keys
        new_state_dict = {}
        for key, value in state_dict.items():
            if 'image_encoder' in key:
                # 移除'image_encoder.'前缀
                new_key = key.replace('image_encoder.', '')
                new_state_dict[new_key] = value

        # 加载到base_encoder
        msg = base_encoder.load_state_dict(new_state_dict)
        print(f"Missing keys: {msg.missing_keys}")
        print(f"Unexpected keys: {msg.unexpected_keys}")

    def _should_use_moe(self, stage_idx, block_idx, stage_length):
        """决定某一层是否应该使用MoE"""
        # 1. stage的最后一层使用MoE
        # if block_idx == stage_length - 1:
        #     return True

        # 2. stage的第一层使用MoE（用于特征转换）
        if stage_idx >= 2 and block_idx == 0:
            return True

        # 3. 在较深的stage中增加MoE使用频率
        # if stage_idx >= 2:  # 在更深的stage
        #     # 每隔一定间隔使用MoE
        #     return block_idx % 2 == 0

        # 4. 浅层stage较少使用MoE
        return False

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = hw
        window_embed = self.trunk.pos_embed_window
        pos_embed = F.interpolate(self.trunk.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def prepare_for_training(self, train_gating=True):
        """准备模型训练,冻结基础参数,只训练LoRA部分"""
        # 冻结所有参数
        for param in self.parameters():
            param.requires_grad = False

        # 只启用LoRA参数和Gating参数
        for module in self.modules():
            if isinstance(module, LoRALayer):
                module.lora_A.requires_grad = True
                module.lora_B.requires_grad = True

            if train_gating and isinstance(module, BaseGatingNetwork):
                for param in module.parameters():
                    param.requires_grad = True

    def print_trainable_parameters(self):
        """打印可训练参数信息"""
        print("Trainable parameters:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.shape}")

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")

    def get_lora_state_dict(self):
        """获取LoRA参数状态字典"""
        lora_state_dict = {}
        for name, module in self.named_modules():
            if isinstance(module, LoRALayer):
                lora_state_dict[f"{name}.lora_A"] = module.lora_A.data
                lora_state_dict[f"{name}.lora_B"] = module.lora_B.data
            elif isinstance(module, BaseGatingNetwork):
                for param_name, param in module.named_parameters():
                    lora_state_dict[f"{name}.{param_name}"] = param.data
        return lora_state_dict

    def load_lora_state_dict(self, state_dict):
        """加载LoRA参数状态字典"""
        for name, module in self.named_modules():
            if isinstance(module, LoRALayer):
                if f"{name}.lora_A" in state_dict:
                    module.lora_A.data = state_dict[f"{name}.lora_A"]
                if f"{name}.lora_B" in state_dict:
                    module.lora_B.data = state_dict[f"{name}.lora_B"]
            elif isinstance(module, BaseGatingNetwork):
                for param_name, param in module.named_parameters():
                    if f"{name}.{param_name}" in state_dict:
                        param.data = state_dict[f"{name}.{param_name}"]

    def forward(self, x, text=None):
        group_id = 0
        if text is not None:
            text_embedding = self.text_encoder.encode(text)
            for idx, organs in ORGAN_GROUPS.items():
                # 检查句子中是否包含该分组的任意器官
                if any(organ in text for organ in organs):
                    group_id = idx
        else:
            text_embedding = None
        x = self.trunk.patch_embed(x)
        H, W = x.shape[1:3]
        x = x + self._get_pos_embed((H, W))

        outputs = []
        losses = [0, 0, 0]  # [total_loss, cls_loss, balance_loss]
        num = 0
        for i, blk in enumerate(self.blocks):
            if isinstance(blk, MoEBlock):
                x, losses_i = blk(x, text_embedding, torch.tensor([group_id]).to(text_embedding.device))
                losses = [l + l_i for l, l_i in zip(losses, losses_i)]
                num += 1
            else:
                x = blk(x)
            if (i == self.stage_ends[-1]) or (i in self.stage_ends):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)
        loss = tuple(l / num if num > 0 else 0.0 for l in losses)

        features, pos = self.neck(outputs)
        if self.scalp > 0:
            features, pos = features[: -self.scalp], pos[: -self.scalp]
        del outputs  # 释放outputs
        src = features[-1]
        output = {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
            "moe_loss": loss # [total_loss, cls_loss, balance_loss]
        }
        return output


class LoRAHieraDet(nn.Module):
    def __init__(self, base_encoder, img_size=512, lora_rank=4, pretrained_model='./checkpoints/sam2_hiera_tiny.pt'):
        super().__init__()
        self.load_base_encoder_weights(base_encoder, pretrain_path=pretrained_model)
        self.trunk = base_encoder.trunk
        self.neck = base_encoder.neck
        self.scalp = base_encoder.scalp
        self.stages = self.trunk.stages
        self.dims = self.trunk.channel_list[::-1]
        self.num_stages = len(self.stages)
        self.stage_ends = [sum(self.stages[:i]) - 1 for i in range(1, len(self.stages) + 1)]

        # Transform original blocks to LoRA blocks
        self.blocks = nn.ModuleList()
        for block in self.trunk.blocks:
            lora_block = ExpertBlock(block, rank=lora_rank, add_baseres=True)
            self.blocks.append(lora_block)

    def load_base_encoder_weights(self, base_encoder, pretrain_path):
        """
        加载预训练权重到base_encoder

        Args:
            base_encoder: 基础编码器模型
            pretrain_path: 预训练权重文件路径
        """
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        state_dict = checkpoint['model']
        # 只保留包含image_encoder的权重，并处理keys
        new_state_dict = {}
        for key, value in state_dict.items():
            if 'image_encoder' in key:
                # 移除'image_encoder.'前缀
                new_key = key.replace('image_encoder.', '')
                new_state_dict[new_key] = value

        # 加载到base_encoder
        msg = base_encoder.load_state_dict(new_state_dict)
        print(f"Missing keys: {msg.missing_keys}")
        print(f"Unexpected keys: {msg.unexpected_keys}")

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = hw
        window_embed = self.trunk.pos_embed_window
        pos_embed = F.interpolate(self.trunk.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def prepare_for_training(self):
        """Prepare model for training by freezing base parameters and enabling only LoRA parameters"""
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        # Enable only LoRA parameters
        for module in self.modules():
            if isinstance(module, LoRALayer):
                module.lora_A.requires_grad = True
                module.lora_B.requires_grad = True

    def print_trainable_parameters(self):
        """Print trainable parameter information"""
        print("Trainable parameters:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.shape}")

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")

    def get_lora_state_dict(self):
        """Get LoRA parameters state dict"""
        lora_state_dict = {}
        for name, module in self.named_modules():
            if isinstance(module, LoRALayer):
                lora_state_dict[f"{name}.lora_A"] = module.lora_A.data
                lora_state_dict[f"{name}.lora_B"] = module.lora_B.data
        return lora_state_dict

    def load_lora_state_dict(self, state_dict):
        """Load LoRA parameters state dict"""
        for name, module in self.named_modules():
            if isinstance(module, LoRALayer):
                if f"{name}.lora_A" in state_dict:
                    module.lora_A.data = state_dict[f"{name}.lora_A"]
                if f"{name}.lora_B" in state_dict:
                    module.lora_B.data = state_dict[f"{name}.lora_B"]

    def forward(self, x):
        x = self.trunk.patch_embed(x)
        H, W = x.shape[1:3]
        x = x + self._get_pos_embed((H, W))

        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i == self.stage_ends[-1]) or (i in self.stage_ends):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)

        features, pos = self.neck(outputs)
        if self.scalp > 0:
            features, pos = features[: -self.scalp], pos[: -self.scalp]
        del outputs  # Free outputs
        src = features[-1]
        output = {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }
        return output
