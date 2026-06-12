import torch
import torch.nn as nn
from DataGenerator import get_testing_dataloader # Bộ sinh dữ liệu đã viết ở bước trước
from MoELayer import MoELayer

# --- THIẾT LẬP THAM SỐ KIỂM THỬ ---
VOCAB_SIZE = 1000
SEQ_LENGTH = 16
D_MODEL = 64
HIDDEN_DIM = 128
NUM_EXPERTS = 4

ALPHA = 0.05     
DES_RATE = 0.95  
UPDATE_METHOD = 3   

# --- KHỞI TẠO MÔ HÌNH NGÔN NGỮ THU NHỎ ---
class MoELanguageModelForTest(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.moe = MoELayer(D_MODEL, HIDDEN_DIM, NUM_EXPERTS, ALPHA, DES_RATE, UPDATE_METHOD)
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE)
        
    def forward(self, idx):
        x = self.embedding(idx)
        x, router_logits = self.moe(x)
        logits = self.lm_head(x)
        return logits, router_logits

# Khởi tạo mô hình và lấy 1 batch dữ liệu
model = MoELanguageModelForTest()
dataloader = get_testing_dataloader(num_samples=10, seq_length=SEQ_LENGTH, vocab_size=VOCAB_SIZE, batch_size=2)
x_batch, y_batch = next(iter(dataloader))

print("==================================================")
print("  RUNNING MOE TEST SUITE...")
print("==================================================")

# ------------------------------------------------
# TEST 1: KIỂM TRA GRADIENT FLOW
# ------------------------------------------------
print("\nTEST 1: Kiểm tra Gradient Flow...")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

logits, router_logits = model(x_batch)
loss = criterion(logits.view(-1, VOCAB_SIZE), y_batch.view(-1))
loss.backward()

# Kiểm tra xem router có nhận được gradient không
router_grad = model.moe.router.weight.grad
if router_grad is not None and torch.sum(torch.abs(router_grad)) > 0:
    print("Thành công: Gradient truyền về Router mượt mà, không bị đứt đoạn!")
else:
    print("Thất bại: Router không nhận được gradient. Hãy kiểm tra lại.")

# ------------------------------------------------
# TEST 2: KIỂM TRA ĐỘ LINH HOẠT CỦA THRESHOLD (Ý tưởng nghiên cứu)
# ------------------------------------------------
print("\nTEST 2: Kiểm tra hành vi định tuyến Adaptive...")
with torch.no_grad():
    model.eval()
    # Gọi trực tiếp hàm select_expert để phân tích mask
    # Giả lập một router_logits ngẫu nhiên để test
    test_flat = model.embedding(x_batch).view(-1, D_MODEL)
    r_logits = model.moe.router(test_flat)
    p_sorted, _ = torch.sort(torch.softmax(r_logits, dim=-1), dim=-1, descending=True)
    
    selected_mask, _ = model.moe.select_expert(p_sorted, test_flat.size(0), ALPHA, DES_RATE)
    
    # Tính số lượng expert trung bình được chọn cho mỗi token
    experts_per_token = selected_mask.sum(dim=-1).float()
    avg_experts = experts_per_token.mean().item()
    min_experts = experts_per_token.min().item()
    max_experts = experts_per_token.max().item()
    
    print(f"Thống kê số lượng Expert được chọn trên mỗi Token:")
    print(f"   - Trung bình: {avg_experts:.2f} experts/token")
    print(f"   - Ít nhất:    {min_experts} expert")
    print(f"   - Nhiều nhất: {max_experts} experts")
    
    if min_experts != max_experts:
        print("Thành công: Thuật toán Adaptive Threshold hoạt động đúng! Các token khác nhau kích hoạt số lượng expert khác nhau.")
    else:
        print("Cảnh báo: Tất cả token đều kích hoạt cùng số lượng expert. Bạn nên điều chỉnh lại tham số ALPHA hoặc DES_RATE.")

# ------------------------------------------------
# TEST 3: ÉP HỌC VẸT (OVERFIT 1 BATCH)
# ------------------------------------------------
print("\nTEST 3: Kiểm tra khả năng hội tụ (Overfit 1 Batch)...")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2) # Tăng LR cho nhanh hội tụ

initial_loss = loss.item()
print(f"   - Loss ở Epoch đầu tiên: {initial_loss:.4f}")

for epoch in range(60):
    optimizer.zero_grad()
    logits, _ = model(x_batch)
    loss = criterion(logits.view(-1, VOCAB_SIZE), y_batch.view(-1))
    loss.backward()
    optimizer.step()

final_loss = loss.item()
print(f"   - Loss ở Epoch thứ 60:  {final_loss:.4f}")

if final_loss < initial_loss * 0.1:
    print("Thành công: Mạng MoE có khả năng tối ưu hóa tốt. Loss đã giảm mạnh!")
else:
    print("Thất bại: Mô hình không thể học thuộc lòng dù chỉ 1 batch. Khả năng cao có lỗi logic tính toán.")
print("\n==================================================")