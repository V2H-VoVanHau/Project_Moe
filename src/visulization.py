import matplotlib.pyplot as plt
import numpy as np

def plot_expert_distribution(percentages, title="Phân bổ số lượng Expert được kích hoạt"):
    """
    Hàm vẽ biểu đồ cột trực quan hóa tỷ lệ token phân bổ vào các nhóm Expert.
    
    Args:
        percentages (list/array): Mảng chứa phần trăm (ví dụ: [76.0, 14.6, 5.4, 4.0])
        title (str): Tiêu đề của biểu đồ
    """
    # 1. Chuẩn bị dữ liệu trục X (Nhãn dán)
    num_experts = len(percentages)
    labels = [f'{i+1} Expert{"s" if i > 0 else ""}' for i in range(num_experts)]
    
    # 2. Khởi tạo khung biểu đồ
    plt.figure(figsize=(8, 6))
    
    # Bảng màu
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    
    # Vẽ các cột
    bars = plt.bar(labels, percentages, color=colors[:num_experts], width=0.6, edgecolor='black', linewidth=1.2)
    
    # 3. Tinh chỉnh chữ
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Số lượng Expert được gán cho mỗi Token', fontsize=12, labelpad=10)
    plt.ylabel('Tỷ lệ Token (%)', fontsize=12, labelpad=10)
    
    plt.ylim(0, max(percentages) + 15)
    
    # Thêm lưới ngang
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    
    for bar in bars:
        bar.set_zorder(3)
        
    # 4. Thêm phần trăm lên đỉnh mỗi cột
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, 
                 yval + 1.5,   # Nâng text lên trên đỉnh cột một chút
                 f'{yval:.1f}%', 
                 ha='center', 
                 va='bottom', 
                 fontsize=11, 
                 fontweight='bold',
                 color='black')
                 
    # 5. Căn lề và hiển thị
    plt.tight_layout()
    
    # plt.savefig('expert_distribution.png', dpi=300, bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    
    data = [52.7, 16.4, 10.9, 20]

    plot_expert_distribution(data)