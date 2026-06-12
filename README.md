# Project_Moe

## I. Mục tiêu

- **Tính khả thi:** Kiểm chứng khả năng áp dụng thực tế của phương pháp chọn top expert thông qua threshold động (xử lý trên kết quả của hàm softmax).
- **Đánh giá hiệu quả:** So sánh 3 phương pháp cập nhật threshold: giảm tuyến tính, giảm theo vị trí và giảm theo tỷ lệ.
- **Bảo toàn logic:** Đảm bảo việc tích hợp cơ chế này không làm phá vỡ cấu trúc và logic tổng thể của mô hình MoE.

## II. Chi tiết Triển khai
**Ngôn ngữ sử dụng:** Python

### 1. Bộ sinh dữ liệu (Data Generator)

- **Cơ chế sinh:** Bỏ token cũ nhất và chèn token mới vào cuối vector, giúp mô hình học mối quan hệ của các cặp từ thay vì các từ đơn lẻ độc lập.
- **Định dạng đầu ra:** Các vector được gộp theo `Batch_Size` tạo thành ma trận 2 chiều `Batch_Size * Sequence_Length`.
- **Cấu hình thử nghiệm:** * `VOCAB_SIZE` = 1000`SEQ_LENGTH` = 32
- `BATCH_SIZE` = 4
- `NUM_SAMPLES` = 100
- *(Đầu ra: Ma trận kích thước 4 * 32)*

- **Lưu ý:** Dữ liệu hoàn toàn ngẫu nhiên và không có quan hệ ngữ nghĩa. Phục vụ duy nhất cho mục đích kiểm tra luồng logic của hệ thống, không dùng để đánh giá độ chính xác của phân phối xác suất router thực tế.

### 2. Tầng MoELayer & Cơ chế Routing

- **Thiết kế:** Sử dụng cơ chế routing với threshold co giãn dựa trên phân phối xác suất của từng token (thay thế cho cơ chế Top-K cố định).
- **Luồng dữ liệu:** Ma trận đầu vào 3 chiều (`Batch_Size * Sequence_Length * D_Model`) $\rightarrow$ Làm phẳng thành 2 chiều $\rightarrow$ Linear Router $\rightarrow$ Masking $\rightarrow$ Expert MLP $\rightarrow$ Định dạng lại về 3 chiều.
- **Cấu hình thử nghiệm:**`D_MODEL` = 64
- `HIDDEN_DIM` = 128
- `NUM_EXPERTS` = 4
- `ALPHA` = 0.05
- `DES_RATE` = 0.95
- `UPDATE_METHOD` = 3

- **Lưu ý:** Tham số `des_rate` phụ thuộc chặt chẽ vào phương pháp cập nhật (`update_method`). Ví dụ: `des_rate = 0.9` ở Method 3 có thể khiến Method 1 bị kẹt ở mức Top 1 expert.

### 3. Quy trình Kiểm thử Hệ thống
Hệ thống được đưa vào một mô hình ngôn ngữ thu nhỏ (`MoELanguageModelForTest`) bao gồm: Tầng Embedding, Tầng định tuyến trung tâm (MoELayer), và Tầng đầu ra (Linear LM Head) để thực hiện 3 bài test:

1. **Kiểm tra Dòng chảy Đạo hàm (Gradient Flow):** Xác minh tính liền mạch của đồ thị tính toán. Ma trận `router.weight.grad` phải khác 0 sau khi gọi `loss.backward()`, chứng minh router tự động cập nhật được trọng số.
2. **Kiểm tra Hành vi Định tuyến Co giãn (Adaptive Routing Behavior):** Đo lường lượng expert được gán cho token ở trạng thái chưa huấn luyện (untrained state). Đảm bảo cơ chế linh hoạt, không bị tình trạng mọi token đều chọn số expert giống hệt nhau.
3. **Kiểm tra Khả năng Hội tụ Cực đại (Overfit a Single Batch):** Ép mô hình học thuộc 1 batch lặp lại trong 60 epochs với learning rate lớn. Hàm mất mát (loss) sụt giảm mạnh chứng minh hệ thống không bị lỗi logic toán học cơ bản.

### 4. Thiết lập Kiểm tra Tuning Tham số

- **Kịch bản:** Sử dụng hạt giống ngẫu nhiên (random seed) cố định tạo ra phân phối xác suất giống hệt nhau cho 1000 tokens mô phỏng.
- **Mục tiêu đo lường:** Số expert trung bình/token, phạm vi co giãn (Min/Max), và phổ phân bổ mật độ.
- **Các phương pháp đối chiếu:****Method 1:** Trừ tuyến tính cố định.
- **Method 2:** Trừ tuyến tính tăng tốc.
- **Method 3:** Suy giảm hình học / Nhân tỷ lệ.

## III. Kết quả

### 1. Bài kiểm tra Logic Hệ thống

- **Vượt qua thành công cả 3 bài test** (Gradient Flow, Adaptive Behavior, Overfit).
- **Vấn đề ghi nhận:** Ở thiết lập gốc, lượng expert/token trung bình đạt 3.62, cho thấy thuật toán đang hơi "bao dung". Cần tinh chỉnh lại bộ tham số.

### 2. Bài kiểm tra Tuning & Đối chiếu
**Cấu hình:** 1000 Tokens | 4 Experts | Alpha gốc: 0.03

| Chỉ số | Method 1 (Trừ cố định) | Method 2 (Trừ tăng tốc) | Method 3 (Nhân tỷ lệ) |
|--------|------------------------|------------------------|----------------------|
| **Des_rate** | 0.02 | 0.01 | 0.70 |
| **Avg Expert/Token** | 2.14 | 2.13 | 1.98 |
| **Phạm vi co giãn** | 1 đến 4 Experts | 1 đến 4 Experts | 1 đến 4 Experts |
| **Tỷ lệ chọn 1 Expert** | 52.7% | 52.7% | 52.7% |
| **Tỷ lệ chọn 2 Experts** | 11.4% | 11.4% | 16.4% |
| **Tỷ lệ chọn 3 Experts** | 5.0% | 6.0% | 10.9% |
| **Tỷ lệ chọn 4 Experts** | 30.9% | 29.9% | 20.0% |
### 3. Phân tích & Đánh giá

- **Hiệu suất thưa thớt (Sparsity):** Toàn bộ hệ thống đạt trạng thái cân bằng lý tưởng. Cả 3 phương pháp đều duy trì lượng chuyên gia ở mức xấp xỉ **2 experts/token**, giúp tiết kiệm đáng kể chi phí tính toán.
- **Độ co giãn (Adaptivity):** Thuật toán thành công trong việc phân bổ cân bằng token ở các vòng lặp đầu và tự động dồn trọng tâm vào các expert có xác suất cao ở các vòng lặp sau.
- **Đặc tính Phương pháp:** Method 1 và Method 2 có tính phân cực rất mạnh (tỷ lệ dồn vào 1 hoặc 4 experts chiếm áp đảo). Trong khi đó, **Method 3** mang lại sự phân bổ mượt mà, linh hoạt và tối ưu hơn.