import torch
import torch.nn as nn
from MoELayer import MoELayer 

def run_ablation_study():
    # 1. Cố định seed
    torch.manual_seed(18)
    
    # Kích cỡ mẫu thử lớn (1000 tokens) để số lượng thống kê mang tính chất đại chúng
    N = 1000          
    NUM_EXPERTS = 4
    D_MODEL = 64
    ALPHA = 0.03
    
    # 2. Giả lập một ma trận p_sorted
    mock_logits = torch.randn(N, NUM_EXPERTS)
    p_sorted, _ = torch.sort(torch.softmax(mock_logits, dim=-1), dim=-1, descending=True)
    
    print("="*70)
    print(f"🔬 BÁO CÁO THỬ NGHIỆM ĐỊNH TUYẾN ADAPTIVE MOE")
    print(f"Cấu hình: {N} Tokens | {NUM_EXPERTS} Experts | Alpha gốc: {ALPHA}")
    print("="*70)
    
    # Định nghĩa tên và bản chất toán học của từng Method 
    methods_config = {
        1: {"name": "Trừ cố định", "des_rate": 0.02},
        2: {"name": "Trừ tăng tốc", "des_rate": 0.01},
        3: {"name": "Nhân tỷ lệ", "des_rate": 0.7} 
    }
    
    # Khởi tạo một layer làm gốc
    moe_layer = MoELayer(d_model=D_MODEL, hidden_dim=128, num_experts=NUM_EXPERTS, 
                         alpha=ALPHA, des_rate=0.1, update_method=1)
    
    for method_id, config in methods_config.items():
        # Cập nhật cấu hình động cho Layer
        moe_layer.update_method = method_id
        current_des_rate = config["des_rate"]
        
        # Chạy thuật toán lọc mask
        selected_mask, _ = moe_layer.select_expert(p_sorted, N, ALPHA, current_des_rate)
        
        # 3. TIẾN HÀNH THỐNG KÊ CHI TIẾT
        # Đếm số expert được chọn trên từng token một
        experts_per_token = selected_mask.sum(dim=-1).float()
        
        avg_exp = experts_per_token.mean().item()
        min_exp = int(experts_per_token.min().item())
        max_exp = int(experts_per_token.max().item())
        
        # Tính tỷ lệ phần trăm phân bổ
        distribution_report = []
        for k in range(1, NUM_EXPERTS + 1):
            percentage = (experts_per_token == k).sum().item() / N * 100
            if percentage > 0:
                distribution_report.append(f"{k} Expert: {percentage:.1f}%")
        
        # 4. IN KẾT QUẢ
        print(f"\nMETHOD {method_id}: {config['name']}")
        print(f"- Tham số des_rate áp dụng: {current_des_rate}")
        print(f"- Số lượng Expert trung bình/Token: {avg_exp:.2f}")
        print(f"- Phạm vi co giãn: Từ {min_exp} đến {max_exp} Experts")
        print(f"- Biểu đồ phân bổ mật độ:  { ' | '.join(distribution_report) }")
        print("-" * 70)

if __name__ == "__main__":
    run_ablation_study()