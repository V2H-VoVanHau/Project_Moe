import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class Expert(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, x):
        return self.net(x)
    
class MoELayer(nn.Module):
    def __init__(self, d_model, hidden_dim, num_experts, alpha, des_rate, update_method=1):
        super().__init__()
        self.num_experts = num_experts
        self.alpha = alpha
        self.des_rate = des_rate
        self.update_method = update_method

        self.experts = nn.ModuleList([Expert(d_model, hidden_dim) for _ in range(num_experts)])
        self.router = nn.Linear(d_model, num_experts)

    # Đã sửa: Thêm self, thay thế if-else bằng torch.where để xử lý tensor
    def init_threshold(self, deviation, alpha):
        return torch.where(deviation == 0, torch.zeros_like(deviation), alpha / deviation)
    
    def select_expert(self, p, num_tokens, alpha, des_rate):
        deviation = torch.std(p, dim=-1, unbiased=False, keepdim=True)
        # Đã sửa: Gọi init_threshold với self
        threshold = self.init_threshold(deviation, alpha)

        device = p.device
        selected_mask = torch.zeros(num_tokens, self.num_experts, dtype=torch.bool, device=device)
        selected_mask[:, 0] = True

        active_tokens = torch.ones(num_tokens, dtype=torch.bool, device=device)

        for i in range(1, self.num_experts):
            p_prev = p[:, i-1]
            p_curr = p[:, i]
            diff = p_prev - p_curr

            cond = (diff <= threshold.squeeze(-1)) & active_tokens

            active_tokens = cond

            selected_mask[active_tokens, i] = True

            if self.update_method == 1:
                threshold = torch.clamp(threshold - des_rate, min=0)
            elif self.update_method == 2:
                threshold = torch.clamp(threshold - des_rate * (i + 1), min=0)
            else:
                threshold = threshold * des_rate

        masked_weights = p * selected_mask
        masked_weights = masked_weights / (masked_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Đã sửa: Trả về kết quả
        return selected_mask, masked_weights
            
            
    def forward(self, x):
        # x: [Batch size, Sequence length, d_model]
        B, S, D = x.shape

        x_flat = x.view(-1, D)
        N = x_flat.size(0)

        router_logits = self.router(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1)

        p_sorted, indices_sorted = torch.sort(routing_weights, dim=-1, descending=True)

        # Đã sửa: Bổ sung toàn bộ logic xử lý bị thiếu cho hàm forward
        # 1. Gọi hàm select_expert để lấy mask và trọng số
        selected_mask, masked_weights = self.select_expert(p_sorted, N, self.alpha, self.des_rate)

        # 2. Tạo tensor kết quả
        final_output = torch.zeros_like(x_flat)

        # 3. Phân phối token vào các expert dựa trên mask
        for idx, expert in enumerate(self.experts):
            matched_mask = (indices_sorted == idx) & selected_mask
            
            if matched_mask.any():
                token_idx, sorted_pos = torch.where(matched_mask)
                
                expert_input = x_flat[token_idx]
                expert_output = expert(expert_input)
                
                weight = masked_weights[token_idx, sorted_pos].unsqueeze(-1)
                final_output[token_idx] += expert_output * weight

        return final_output.view(B, S, D), router_logits