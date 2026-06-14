# CHI TIẾT XỬ LÝ VÀ CHUẨN BỊ DỮ LIỆU

Kết hợp 3 bộ dữ liệu (RAF-DB, FER2013, CK+) tạo sự đa dạng lớn do đó cần pipeline tiền xử lý chặt chẽ để giải quyết sự bất đồng nhất.

## 1. Thu thập, Sàng lọc và Chuẩn hóa cấu trúc

-   **Trích xuất nhãn:** Từ 3 dataset gốc, chỉ giữ lại các hình ảnh thuộc 4 nhãn: `Happiness`, `Sadness`, `Anger`, `Neutral`. Loại bỏ các nhãn như Fear, Surprise, Disgust để tránh nhiễu.
    
-   **Chuẩn hóa cấu trúc thư mục:** Quy hoạch toàn bộ tên thư mục của các dataset về chung một bộ Enum chuẩn (`anger`, `happiness`, `neutral`, `sadness`). Ví dụ: Chuyển đổi nhãn dạng số (4, 5, 6, 7) của RAF-DB và nhãn tính từ (angry, happy) của FER-2013 về đúng chuẩn danh từ để thống nhất định dạng đầu vào.
    
-   **Chú ý:** Dữ liệu đầu vào là các bộ đã public, chứa hình ảnh tập trung vào khuôn mặt. Do đó, **bỏ qua bước chạy thuật toán trích xuất khuôn mặt (như Haar Cascades hay MTCNN)** để tiết kiệm tài nguyên xử lý dữ liệu.
    

## 2. Phương pháp Định dạng Dữ liệu (Data Formatting)

Để phục vụ kiến trúc **Dual-Branch GCN-CNN** (kết hợp thông tin kết cấu bề mặt - Texture và thông tin hình học - Geometric), dữ liệu đầu ra được lấy từ thư mục `preprocessed/` và chia thành 2 nhánh dữ liệu song song trong thư mục `processed/`:

**Nhánh 1: Hình ảnh khuôn mặt (Texture Branch)**

-   **Mục đích:** Huấn luyện nhánh CNN để trích xuất đặc trưng bề mặt/kết cấu (ví dụ: nếp nhăn, khóe môi, mắt).
    
-   **Xử lý:** Chuyển toàn bộ ảnh về **ảnh xám (Grayscale - 1 kênh màu)** và Resize về kích thước `224x224` pixels.
    
-   **Lý do:** Biểu cảm khuôn mặt chủ yếu phụ thuộc vào độ tương phản cục bộ và đường nét chứ không phụ thuộc vào màu sắc. Sử dụng ảnh Grayscale giúp giảm 3 lần dung lượng bộ nhớ và chi phí tính toán. Định dạng `224x224` đảm bảo độ phân giải đủ tốt để trích xuất landmark chính xác.
    

**Nhánh 2: Đồ thị đặc trưng hình học (Geometric/Graph Branch)**

-   **Mục đích:** Huấn luyện nhánh GCN để học các đặc trưng cấu trúc/khoảng cách hình học giữa các bộ phận trên khuôn mặt.
    
-   **Xử lý:** Trích xuất tọa độ $(x, y)$ của **468 điểm mốc khuôn mặt (Landmarks)** bằng `MediaPipe Tasks API` (Face Landmarker). Tọa độ được lưu dưới dạng file ma trận `.npy` có kích thước `(468, 2)` tương ứng với từng ảnh.
    
-   **Quy tắc đồng bộ (Crucial Deletion Rule):** Nếu ảnh không phát hiện được khuôn mặt/landmark thông qua MediaPipe, ảnh nguồn đó sẽ **bị xóa hoàn toàn** khỏi thư mục `processed/images/` để đảm bảo sự đồng bộ 100% giữa ảnh và đồ thị.
    

## 3. Gom Pool, Phân chia và Cân bằng Dữ liệu

Toàn bộ các cặp dữ liệu (ảnh + đồ thị) sau khi xử lý định dạng sẽ gộp chung vào một "Pool" duy nhất trước khi phân chia lại để tránh sự lệch pha giữa các bộ dữ liệu gốc.

-   **Phân chia tập dữ liệu:** Sử dụng tỷ lệ `70% Training` - `15% Validation` - `15% Testing` sử dụng phân tầng (Stratified) dựa trên nhãn cảm xúc.
    
-   **Cân bằng lớp trên tập Training:** Tập dữ liệu gốc bị lệch nặng về lớp `happiness`. Để mô hình không bị thiên vị, trên **tập Training**, thực hiện cân bằng lớp về đúng **4,500 mẫu cho mỗi lớp** (Undersampling đối với các lớp đa số như happiness/neutral, và Oversampling bằng cách nhân bản đối với các lớp thiểu số như anger/sadness).
    
-   **Đồng bộ hóa dữ liệu final:** Quá trình Undersampling/Oversampling phải được thực hiện đồng thời và đồng bộ cho cả file ảnh (`.jpg`/`.png`) và file đồ thị (`.npy`). Ví dụ: Nếu cặp ảnh `img_001.jpg` được nhân bản thành `img_001_dup1.jpg`, thì file đồ thị `img_001.npy` tương ứng cũng phải được sao chép thành `img_001_dup1.npy`.

## 4. Data Augmentation (Tăng cường Dữ liệu)

Do đặc thù của kiến trúc hai nhánh (với đồ thị landmark được trích xuất tĩnh trước khi đưa vào DataLoader):
-   **KHÔNG áp dụng** các phép biến đổi hình học làm thay đổi vị trí không gian của khuôn mặt (như xoay, lật, dịch chuyển ảnh) trong DataLoader nếu không cập nhật đồng bộ ma trận landmark tương ứng.
-   **Áp dụng Augmentation phi hình học:** Chỉ áp dụng các phép biến đổi về cường độ pixel/ánh sáng (ví dụ: tăng giảm ngẫu nhiên độ sáng - Random Brightness) trực tiếp trên ảnh trong Custom DataLoader. Các phép biến đổi này giúp mô hình tăng khả năng chống chịu nhiễu ánh sáng mà hoàn toàn không ảnh hưởng tới tọa độ hình học của các landmark trong nhánh đồ thị.