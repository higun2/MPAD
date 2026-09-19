import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
from torch import Tensor

class Attention3D_TextKV(nn.Module):
    """
    Cross attention where the query is a 3D input (B, C, D, H, W) and the key and value are 1D text inputs (B, embed_dim).
    """
    def __init__(
        self,
        embedding_dim: int,  # can be matched to the query channel count (C)
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide internal_dim."

        # Query projection from the 3D input
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        # Key and value projection from the 1D text input
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        # Final output projection
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
    
    def _separate_heads(self, x: Tensor) -> Tensor:
        """
        x: (B, N, C) -> (B, num_heads, N, C_per_head)
        """
        B, N, C = x.shape
        x = x.view(B, N, self.num_heads, C // self.num_heads)
        return x.transpose(1, 2)
    
    def _recombine_heads(self, x: Tensor) -> Tensor:
        """
        x: (B, num_heads, N, C_per_head) -> (B, N, C)
        """
        B, num_heads, N, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(B, N, num_heads * c_per_head)
    
    def forward(self, q: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """
        q: Query tensor, shape (B, C, D, H, W)
        key: Key tensor (1D text), shape (B, embed_dim)
        value: Value tensor (1D text), shape (B, embed_dim)
        """
        B, C, D, H, W = q.shape
        N = D * H * W  # spatial tokens
        
        # Flatten query: (B, C, D, H, W) -> (B, N, C)
        q_seq = q.view(B, C, N).transpose(1, 2)  # (B, N, C)
        
        # Query projection: (B, N, embedding_dim) -> (B, N, internal_dim)
        q_proj = self.q_proj(q_seq)  # (B, N, internal_dim)
        
        # Key and value are 1D inputs, so expand the sequence length to 1: (B, 1, embed_dim)
        key_seq = key.unsqueeze(1)
        value_seq = value.unsqueeze(1)
        
        # Key/Value projection: (B, 1, embed_dim) -> (B, 1, internal_dim)
        k_proj = self.k_proj(key_seq)
        v_proj = self.v_proj(value_seq)
        
        # Separate heads
        q_heads = self._separate_heads(q_proj)  # (B, num_heads, N, C_per_head)
        k_heads = self._separate_heads(k_proj)    # (B, num_heads, 1, C_per_head)
        v_heads = self._separate_heads(v_proj)    # (B, num_heads, 1, C_per_head)
        
        # Attention scores: (B, num_heads, N, 1)
        _, _, _, c_per_head = q_heads.shape
        scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / math.sqrt(c_per_head)
        attn_weights = F.softmax(scores, dim=-1)  # (B, num_heads, N, 1)
        
        # Weighted sum: (B, num_heads, N, C_per_head)
        attn_output = torch.matmul(attn_weights, v_heads)
        
        # Recombine heads: (B, N, internal_dim)
        attn_output = self._recombine_heads(attn_output)
        
        # Out projection: (B, N, embedding_dim)
        attn_output = self.out_proj(attn_output)
        
        # Reshape back to 3D: (B, N, embedding_dim) -> (B, embedding_dim, D, H, W)
        attn_output = attn_output.transpose(1, 2).view(B, C, D, H, W)
        
        # Residual connection: query + attention output (optional)
        return q + attn_output
    
    
class ResNetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, norm_layer=True):
        super(ResNetBlock3D, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm1 = nn.InstanceNorm3d(out_channels) if norm_layer else nn.Identity()

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding)
        self.norm2 = nn.InstanceNorm3d(out_channels) if norm_layer else nn.Identity()

        # Activation
        self.activation = nn.LeakyReLU(0.2)

        # Skip connection
        self.skip = None
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.InstanceNorm3d(out_channels) if norm_layer else nn.Identity()
            )
        self.skip_scale = nn.Parameter(torch.tensor(1.0))  # Skip scaling factor

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)  # Activation

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out)  # Activation

        if self.skip is not None:
            identity = self.skip(identity)
        
        out += self.skip_scale * identity  # Scaled skip connection
        return out

class FeedForward(nn.Module):
    """FeedForward Network for MoE Experts"""
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    
class AdaIN(nn.Module):
    def __init__(self, feature_dim, num_channels):
        super().__init__()
        self.mlp_gamma = nn.Linear(feature_dim, num_channels)
        self.mlp_beta = nn.Linear(feature_dim, num_channels)

    def forward(self, x, clip_features):
        mean_x = x.mean(dim=[2, 3, 4], keepdim=True)
        std_x = x.std(dim=[2, 3, 4], keepdim=True) + 1e-5
        
        gamma = self.mlp_gamma(clip_features)[:, :, None, None, None]  
        beta = self.mlp_beta(clip_features)[:, :, None, None, None]  
        return gamma * (x - mean_x) / std_x + beta

class SwitchGate(nn.Module):
    def __init__(self, dim, num_experts: int, capacity_factor: float = 1.0, epsilon: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.epsilon = epsilon
        self.w_gate = nn.Linear(dim, num_experts)

    def forward(self, x: torch.Tensor, use_aux_loss=False, gate_training=False):
        if gate_training:
            return self.w_gate(x)
        gate_scr = F.softmax(self.w_gate(x), dim=-1)
        top_k_scores, top_k_indices = gate_scr.topk(1, dim=-1)
        
        mask = torch.zeros_like(gate_scr).scatter_(1, top_k_indices, 1)
        masked_gate_scores = gate_scr * mask

        denominators = masked_gate_scores.sum(0, keepdim=True) + self.epsilon
        gate_scores = (masked_gate_scores / denominators) * int(self.capacity_factor * x.size(0))

        if use_aux_loss:
            load = gate_scores.sum(0)
            importance = gate_scores.sum(1)
            loss = ((load.unsqueeze(0) - importance.unsqueeze(1)) ** 2).mean()
            return gate_scores, loss*0.2
        return gate_scores, None


class AdaINResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, feature_dim, num_experts):
        super().__init__()
        self.learned_shortcut = (in_channels != out_channels)
        mid_channels = min(in_channels, out_channels)

        self.conv_0 = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if self.learned_shortcut:
            self.conv_s = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        
        self.adain_0 = AdaIN(feature_dim, mid_channels)
        self.adain_1 = AdaIN(feature_dim, out_channels)
        
    
    def forward(self, x, clip_features):
        x_s = self.shortcut(x)
        dx = self.conv_0(F.leaky_relu(self.adain_0(x, clip_features), 0.2))
        dx = self.conv_1(dx)
        dx = self.conv_2(F.leaky_relu(self.adain_1(dx, clip_features), 0.2))
        return x_s + dx
    
    def shortcut(self, x):
        return self.conv_s(x) if self.learned_shortcut else x

# class SwitchAdaINGenerator(nn.Module):
#     def __init__(self, z_dim=3, feature_dim=512, num_experts=4):
#         super().__init__()
#         nf = 64

#         self.linear_1 = nn.Linear(feature_dim, feature_dim)
#         self.linear_2 = nn.Linear(feature_dim, feature_dim)
#         self.in_block = AdaINResnetBlock(z_dim, nf, feature_dim, num_experts)
#         self.out_block = AdaINResnetBlock(nf, nf, feature_dim, num_experts)

#         self.conv_in = nn.Sequential(
#             nn.Conv3d(z_dim, z_dim, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.InstanceNorm3d(z_dim)
#         )
#         self.conv_mid = nn.Conv3d(nf, nf, kernel_size=3, padding=1)
#         self.conv_out = nn.Conv3d(nf, z_dim, kernel_size=3, padding=1)
    
#     def forward(self, x, clip_features):
#         clip_features = self.linear_1(clip_features)
#         clip_features = self.linear_2(clip_features)
#         x = self.conv_in(x)
        
#         x = self.in_block(x, clip_features)
#         x = self.conv_mid(x)
#         x = self.out_block(x, clip_features)
#         x = self.conv_out(x)
#         return x

class SwitchAdaINGenerator(nn.Module):
    def __init__(self, z_dim=3, feature_dim=512, num_experts=4):
        super().__init__()
        nf = 64

        self.linear_1 = nn.Linear(feature_dim, feature_dim)
        self.linear_2 = nn.Linear(feature_dim, nf//2)
        # self.in_block = AdaINResnetBlock(z_dim, nf, feature_dim, num_experts)
        # self.out_block = AdaINResnetBlock(nf, nf, feature_dim, num_experts)

        self.conv_in = nn.Sequential(
            nn.Conv3d(z_dim, nf//2, kernel_size=3, padding=1),
            # nn.LeakyReLU(0.2),
            nn.InstanceNorm3d(nf//2)
        )
        # self.conv_mid = nn.Conv3d(nf, nf, kernel_size=3, padding=1)
        self.res_block1 = ResNetBlock3D(nf, nf)
        self.res_block2 = ResNetBlock3D(nf, nf)
        self.res_block3 = ResNetBlock3D(nf, nf)
        
        self.res_block4 = ResNetBlock3D(nf+nf//2, nf)
        self.res_block5 = ResNetBlock3D(nf, nf+nf//2)
        # self.res_block6 = ResNetBlock3D(nf, nf)
        
        self.conv_out = nn.Sequential(
            nn.Conv3d(nf+nf//2, nf, kernel_size=3, padding=1),
            nn.Conv3d(nf, z_dim, kernel_size=3, padding=1)
        )
        
    
    def forward(self, x, src_clip_features, tar_clip_features):
        tar_clip_features = self.linear_1(tar_clip_features)
        tar_clip_features = self.linear_2(tar_clip_features)
        
        tar_clip_features = tar_clip_features.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        tar_clip_features = tar_clip_features.repeat(1, 1, x.size(2), x.size(3), x.size(4))
        
        src_clip_features = self.linear_1(src_clip_features)
        src_clip_features = self.linear_2(src_clip_features)
        src_clip_features = src_clip_features.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        src_clip_features = src_clip_features.repeat(1, 1, x.size(2), x.size(3), x.size(4))
        
        x = self.conv_in(x)
        x = torch.cat((x, src_clip_features), dim=1)
        # x = self.in_block(x, clip_features)
        # x = self.conv_mid(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        # x = self.out_block(x, clip_features)
        x = torch.cat((x, tar_clip_features), dim=1)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.conv_out(x)
        return x

class DualCrossAttentionGenerator(nn.Module):
    def __init__(self, z_dim=3, feature_dim=512, nf=128):
        super().__init__()
        
        self.text_linear = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, nf)
        )
        
        self.conv_in = nn.Sequential(
            nn.Conv3d(z_dim, nf, kernel_size=3, padding=1),
            nn.InstanceNorm3d(nf),
            nn.LeakyReLU(0.2)
        )
        
        self.cross_attn_src = Attention3D_TextKV(embedding_dim=nf, num_heads=8)
        
        self.res_block1 = ResNetBlock3D(nf, nf)
        self.res_block2 = ResNetBlock3D(nf, nf)
        
        self.cross_attn_tar = Attention3D_TextKV(embedding_dim=nf, num_heads=8)
        
        self.res_block3 = ResNetBlock3D(nf, nf)
        self.res_block4 = ResNetBlock3D(nf, nf)

        
        self.conv_out = nn.Sequential(
            nn.Conv3d(nf, nf//2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(nf//2, z_dim, kernel_size=3, padding=1)
        )
        
    def forward(self, x, src_clip_features, tar_clip_features):
        src_text = self.text_linear(src_clip_features)  # (B, nf)
        tar_text = self.text_linear(tar_clip_features)  # (B, nf)
        
        x = self.conv_in(x)  # (B, nf, D, H, W)
        x = self.cross_attn_src(x, src_text, src_text)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.cross_attn_tar(x, tar_text, tar_text)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.conv_out(x)
        return x 

# class SwitchMoEGenerator(nn.Module):
#     def __init__(self, z_dim=3, feature_dim=512, num_experts=4, capacity_factor=1.0):
#         super().__init__()
#         self.z_dim = z_dim
#         self.feature_dim = feature_dim
#         self.num_experts = num_experts
#         self.capacity_factor = capacity_factor

#         # Each expert is now a full SwitchAdaINGenerator
#         self.experts = nn.ModuleList([SwitchAdaINGenerator(z_dim, feature_dim) for _ in range(num_experts)])
#         self.gate = SwitchGate(feature_dim, num_experts, capacity_factor)

#     def forward(self, x, clip_features, target=None, use_aux_loss=False, gate_training=False):
#         if gate_training:
#             gate_loss = self.gate(clip_features, gate_training=gate_training)
#             return gate_loss
        
#         gate_scores, loss = self.gate(clip_features, use_aux_loss)
#         # if target is not None:
#         #     print(f"target: {target}")
#         #     print(f"gate_scores: {gate_scores}")
#         expert_outputs = [expert(x, clip_features) for expert in self.experts]
#         stacked_expert_outputs = torch.stack(expert_outputs, dim=-1)
#         output = torch.sum(gate_scores.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1) * stacked_expert_outputs, dim=-1)
        
#         return output, loss

class SwitchMoEGenerator(nn.Module):
    def __init__(self, z_dim=3, feature_dim=512, num_experts=4, capacity_factor=1.0):
        super().__init__()
        self.z_dim = z_dim
        self.feature_dim = feature_dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor

        # Each expert is now a full SwitchAdaINGenerator
        # self.experts = nn.ModuleList([SwitchAdaINGenerator(z_dim, feature_dim) for _ in range(num_experts)])
        self.experts2 = nn.ModuleList([DualCrossAttentionGenerator(z_dim, feature_dim) for _ in range(num_experts)])
        self.gate = SwitchGate(feature_dim, num_experts, capacity_factor)

    def forward(self, x, src_clip_features, tar_clip_features, target=None, use_aux_loss=False, gate_training=False):
        if gate_training:
            gate_loss = self.gate(src_clip_features, gate_training=gate_training)
            return gate_loss
        
        gate_scores, loss = self.gate(tar_clip_features, use_aux_loss)
        # if target is not None:
        #     print(f"target: {target}")
        #     print(f"gate_scores: {gate_scores}")
        # expert_outputs = [expert(x, src_clip_features, tar_clip_features) for expert in self.experts]
        expert_outputs = [expert(x, src_clip_features, tar_clip_features) for expert in self.experts2]

        stacked_expert_outputs = torch.stack(expert_outputs, dim=-1)
        output = torch.sum(gate_scores.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1) * stacked_expert_outputs, dim=-1)
        
        return output, loss
    
# class HierarchicalSwitchMoEGenerator(nn.Module):
#     def __init__(self, z_dim=3, feature_dim=512, num_moe_experts=4, capacity_factor=1.0):
#         super().__init__()
#         self.z_dim = z_dim
#         self.feature_dim = feature_dim
#         self.num_moe_experts = num_moe_experts
#         self.capacity_factor = capacity_factor

#         # Each MoE generator is a SwitchMoEGenerator
#         self.moe_experts = nn.ModuleList([
#             SwitchMoEGenerator(z_dim, feature_dim, num_moe_experts, capacity_factor)
#             for _ in range(num_moe_experts)
#         ])
        
#         # Gating layer that selects an MoE generator from src_clip_features
#         self.moe_gate = SwitchGate(feature_dim, num_moe_experts, capacity_factor)

#     def forward(self, x, src_clip_features, tar_clip_features, use_aux_loss=False, gate_training=False):
#         if gate_training:
#             # Gating loss for top-level MoE selection
#             moe_gate_loss = self.moe_gate(src_clip_features, gate_training=gate_training)
#             return moe_gate_loss

#         # Select a SwitchMoEGenerator using src_clip_features
#         moe_gate_scores, moe_gate_loss = self.moe_gate(src_clip_features, use_aux_loss)
#         moe_outputs = [moe(x, tar_clip_features) for moe in self.moe_experts]
        
#         stacked_moe_outputs, stacked_moe_losses = zip(*moe_outputs)  # Unpacking (output, loss) pairs
#         stacked_moe_outputs = torch.stack(stacked_moe_outputs, dim=-1)
#         stacked_moe_losses = torch.stack(stacked_moe_losses, dim=-1)
        
#         output = torch.sum(moe_gate_scores.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1) * stacked_moe_outputs, dim=-1)
#         total_loss = torch.sum(moe_gate_scores * stacked_moe_losses, dim=-1)
        
#         return output, moe_gate_loss + total_loss
    

# class SwitchMoEGenerator(nn.Module):
#     def __init__(self, z_dim=3, feature_dim=512, num_experts=4, capacity_factor=1.0):
#         super().__init__()
#         self.z_dim = z_dim
#         self.feature_dim = feature_dim
#         self.num_experts = num_experts
#         self.capacity_factor = capacity_factor

#         # Each expert is now a full SwitchAdaINGenerator
#         self.experts = nn.ModuleList([SwitchAdaINGenerator(z_dim, feature_dim) for _ in range(num_experts)])

#     def forward(self, x, clip_features, gate_scores, loss, use_aux_loss=False, gate_training=False):
#         if gate_training:
#             return self.gate(clip_features, gate_training=gate_training)
        
#         top1_expert_idx = torch.argmax(gate_scores, dim=-1)  # Select top-1 expert
#         selected_expert = self.experts[top1_expert_idx]  # Run only the selected expert
#         output = selected_expert(x, clip_features)
        
#         return output, loss
    
# class HierarchicalSwitchMoEGenerator(nn.Module):
#     def __init__(self, z_dim=3, feature_dim=512, num_moe_experts=4, capacity_factor=1.0):
#         super().__init__()
#         self.z_dim = z_dim
#         self.feature_dim = feature_dim
#         self.num_moe_experts = num_moe_experts
#         self.capacity_factor = capacity_factor

#         # Each MoE Generator is a SwitchMoEGenerator
#         self.moe_experts = nn.ModuleList([
#             SwitchMoEGenerator(z_dim, feature_dim, num_moe_experts, capacity_factor)
#             for _ in range(num_moe_experts)
#         ])
        
#         # Gating layers for both source and target-based selection
#         self.moe_gate = SwitchGate(feature_dim, num_moe_experts, capacity_factor)

#     def forward(self, x, src_clip_features, tar_clip_features, use_aux_loss=False, gate_training=False):
#         if gate_training:
#             return self.moe_gate(src_clip_features, gate_training=gate_training)

#         # Select top-1 SwitchMoEGenerator based on src_clip_features
#         moe_gate_scores, moe_gate_loss = self.moe_gate(src_clip_features, use_aux_loss)
#         top1_moe_idx = torch.argmax(moe_gate_scores, dim=-1)  # Select top-1 MoE expert
#         selected_moe = self.moe_experts[top1_moe_idx]  # Run only the selected MoE
        
#         # Select top-1 internal expert within the selected MoE based on tar_clip_features
#         tar_gate_scores, tar_gate_loss = self.moe_gate(tar_clip_features, use_aux_loss)
#         output, expert_loss = selected_moe(x, tar_clip_features, tar_gate_scores, tar_gate_loss, use_aux_loss=use_aux_loss)
        
#         return output, moe_gate_loss + expert_loss
