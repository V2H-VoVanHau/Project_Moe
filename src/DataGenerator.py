import torch
from torch.utils.data import Dataset, DataLoader

class SyntheticTextDataset(Dataset):
    """
    Bộ dữ liệu giả lập văn bản: Tự động sinh ra các chuỗi Token ID ngẫu nhiên
    và chia tách thành X (đầu vào) và Y (nhãn mục tiêu dịch chuyển 1 token).
    """
    def __init__(self, num_samples, seq_length, vocab_size):
        self.num_samples = num_samples
        self.seq_length = seq_length
        
        # Sinh ma trận ngẫu nhiên các mã số từ (Token IDs) nằm trong khoảng [0, vocab_size)
        # Cần tạo độ dài (seq_length + 1) để khi cắt làm đôi X và Y không bị thiếu từ
        self.data = torch.randint(0, vocab_size, (num_samples, seq_length + 1))
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # Ví dụ chuỗi gốc: [Từ_A, Từ_B, Từ_C, Từ_D]
        # X (Đầu vào):    [Từ_A, Từ_B, Từ_C]
        # Y (Mục tiêu):   [Từ_B, Từ_C, Từ_D]
        x = self.data[idx, :-1]
        y = self.data[idx, 1:]
        return x, y

def get_testing_dataloader(num_samples=200, seq_length=32, vocab_size=1000, batch_size=8, shuffle=True):
    """
    Hàm đóng gói giúp khởi tạo nhanh DataLoader phục vụ kiểm thử
    """
    dataset = SyntheticTextDataset(num_samples, seq_length, vocab_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader