# Tối ưu hóa Mạng Nơ-ron Tích chập (CNN) với Nội suy Parabol trên tập MNIST

**Tác giả:** Dương Thuỷ Tiên

**Môn học:** Toán cho trí tuệ nhân tạo nâng cao

**Mục tiêu bài Lab:** Xây dựng mạng Deep Learning phân loại chữ số viết tay (MNIST) và thực nghiệm thay thế thuật toán cập nhật Gradient Descent (Adadelta) bằng phương pháp tìm kiếm tối ưu không sử dụng Tốc độ học (Learning Rate): **Nội suy Parabol (Parabolic Interpolation)**.
## 1. Kiến trúc Mạng Nơ-ron (CNN Architecture)
Mô hình được thiết kế để trích xuất đặc trưng không gian của ảnh 28x28 thông qua các lớp tích chập, sau đó phân loại bằng các lớp kết nối đầy đủ.

| Lớp (Layer) | Chức năng & Phép toán | Kích thước Output |
| :--- | :--- | :--- |
| **Conv2d (1)** | Lọc đặc trưng cấp thấp (32 filters, kernel 3x3) | 32 x 26 x 26 |
| **ReLU** | Hàm kích hoạt phi tuyến | 32 x 26 x 26 |
| **Conv2d (2)** | Lọc đặc trưng cấp cao (64 filters, kernel 3x3) | 64 x 24 x 24 |
| **MaxPool2d** | Gộp cục bộ, giảm chiều dữ liệu (kernel 2x2) | 64 x 12 x 12 |
| **Dropout (1)** | Điều chuẩn, tắt ngẫu nhiên 25% nơ-ron | 64 x 12 x 12 |
| **Flatten** | Duỗi ma trận 3D thành vector 1D (64x12x12) | **9216** |
| **Linear (1)** | Lớp ẩn kết nối đầy đủ (Fully Connected) | 128 |
| **Dropout (2)** | Tắt ngẫu nhiên 50% nơ-ron chống Overfitting | 128 |
| **Linear (2)** | Lớp đầu ra tương ứng 10 chữ số | 10 |
| **LogSoftmax** | Chuẩn hóa đầu ra thành Log xác suất | 10 |

---

## 2. Cơ chế Cập nhật & Hàm Mất mát (Loss Function)
Mô hình sử dụng hàm **Negative Log-Likelihood Loss (NLLLoss)** kết hợp với đầu ra **LogSoftmax** từ mạng. Về mặt toán học, sự kết hợp này tương đương với việc thực thi tối ưu hóa hàm **Cross-Entropy Loss**, giúp mô hình hội tụ nhanh bằng cách phạt nặng các dự đoán sai lệch có độ tự tin cao.

---

## 3. Tối ưu hóa bằng Nội suy Parabol 
Thay vì sử dụng thuật toán Adadelta với một Learning Rate (Tốc độ học) tĩnh, dự án này áp dụng phương pháp **Nội suy Parabol kế tiếp**.

**Cơ chế hoạt động:**
Thuật toán tận dụng hướng của Gradient làm đường dẫn, nhưng tự động tìm kiếm kích thước bước nhảy tối ưu nhất bằng cách:
1. Đánh giá hàm Loss tại 3 điểm thử nghiệm: Điểm đứng im, điểm nhích nhẹ, và điểm nhích thêm. Thu được 3 giá trị tương ứng L1, L2, L3.
2. Xây dựng một phương trình bậc 2 (Parabol) nội suy đi qua 3 tọa độ này.
3. Áp dụng công thức giải tích đỉnh Parabol để nhảy trực tiếp đến tọa độ cực tiểu địa phương:

$$
w^* = w_2 - \frac{1}{2} \frac{(w_2 - w_1)^2 (L_2 - L_3) - (w_2 - w_3)^2 (L_2 - L_1)}{(w_2 - w_1) (L_2 - L_3) - (w_2 - w_3) (L_2 - L_1)}
$$

Phương pháp này giúp mô hình tự động thích ứng với độ cong của mặt phẳng Loss mà không cần phụ thuộc vào siêu tham số tốc độ học cấu hình bằng tay.

---

## 4. Hướng dẫn chạy chương trình (How to run)

**Yêu cầu môi trường:**
* Python 3.8+
* PyTorch & Torchvision

**Cài đặt thư viện:**
```bash
pip install torch torchvision