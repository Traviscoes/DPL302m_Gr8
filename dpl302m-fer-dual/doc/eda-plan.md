# KẾ HOẠCH PHÂN TÍCH KHÁM PHÁ DỮ LIỆU

Quá trình EDA chia làm 2 giai đoạn:

## GIAI ĐOẠN 1: KHÁM PHÁ DỮ LIỆU NGUỒN

**Mục tiêu:** Hiểu rõ bản chất, sự đa dạng, độ lệch và các đặc tính vật lý của 3 bộ dữ liệu gốc (CK+, FER-2013, RAF-DB).

### 1.1 Phân bố Nhãn

-   **Thao tác:** Đếm tổng số lượng ảnh của từng nhãn (`anger`, `happiness`, `neutral`, `sadness`) và phân tách rõ ràng theo từng dataset.
    
-   **Trực quan hóa (Visualization):**
    -   Sử dụng biểu đồ cột ghép (Grouped Bar Chart) hoặc biểu đồ cột chồng (Stacked Bar Chart). Trục X là 4 nhãn cảm xúc, trục Y là số lượng ảnh, màu sắc đại diện cho 3 dataset.
    -   Vẽ biểu đồ tròn (Pie Chart) thể hiện tỷ trọng đóng góp của mỗi dataset vào tổng pool dữ liệu (Ví dụ: FER-2013 chiếm bao nhiêu %, RAF-DB chiếm bao nhiêu %).
    
-   **Insight:** Chỉ ra được sự mất cân bằng tự nhiên (Happiness thường áp đảo). Đưa ra bằng chứng thuyết phục cho việc tại sao hệ thống cần sử dụng kỹ thuật Undersampling/Oversampling trên tập huấn luyện.
    

### 1.2 Khác biệt Cấu trúc & Độ phân giải (Dimensionality Analysis)

-   **Thao tác:** Quét toàn bộ file ảnh, trích xuất thông tin về chiều rộng (width), chiều cao (height) và số kênh màu (channels).
    
-   **Trực quan hóa:**
    -   Sử dụng biểu đồ phân tán (Scatter Plot): Trục X là Width, trục Y là Height. Các điểm ảnh được tô màu theo Dataset.
    -   Biểu đồ phân phối (KDE/Histogram) cho tỷ lệ khung hình (Aspect Ratio = Width/Height) để xem ảnh có bị méo nhiều không.
-   **Insight:** Sẽ thấy rõ cụm điểm của FER-2013 hội tụ tuyệt đối ở mốc `48x48`.
    -   CK+ và RAF-DB sẽ phân tán rộng hơn ở các độ phân giải cao.
    -   Biện luận cho việc tại sao cần resize về kích thước chuẩn `224x224` để chạy cả hai nhánh (ảnh xám và đồ thị landmark trích xuất qua MediaPipe).

### 1.3 Đánh giá Định tính (Qualitative Visual Inspection)

-   **Thao tác:** Lấy ngẫu nhiên $N$ bức ảnh (ví dụ lưới 4x4) cho mỗi tập dataset, trải đều qua 4 cảm xúc.
    
-   **Trực quan hóa:** In lưới ảnh lên màn hình, hiển thị kèm tên file và kích thước gốc.
    
-   **Insight (Nhận xét bằng mắt thường):**
    -   **CK+:** Sẽ thấy rõ tính chất "Lab-controlled" (chụp trong phòng thí nghiệm): Hậu cảnh trống, ánh sáng đều, người mẫu nhìn thẳng, biểu cảm cường điệu.
    -   **RAF-DB:** Nhận diện rõ tính "In-the-wild": Đa dạng chủng tộc, điều kiện ánh sáng phức tạp, góc nghiêng, có yếu tố che khuất (kính, tay).
    -   **FER-2013:** Ảnh nhỏ, độ phân giải thấp, đôi khi có nhiễu (watermark, ảnh hoạt hình trộn lẫn).

### 1.4 Phân bố Cường độ Sáng (Pixel Intensity & Illumination)

-   **Thao tác:** Chuyển các ảnh mẫu về thang độ xám (Grayscale), dàn phẳng (flatten) ma trận pixel và tính toán tần suất xuất hiện của các giá trị từ 0 đến 255.
    
-   **Trực quan hóa:** Biểu đồ Histogram về cường độ sáng (Pixel Intensity) so sánh giữa 3 bộ dữ liệu.
    
-   **Insight:** Đánh giá xem có bộ dữ liệu nào bị tối quá (skew lệch trái) hay sáng quá (skew lệch phải) không. Điều này củng cố tính tất yếu của việc sử dụng bước Chuẩn hóa (Normalization `1./255`) trước khi đưa vào mô hình học sâu.
    

---

## GIAI ĐOẠN 2: KIỂM ĐỊNH DỮ LIỆU HUẤN LUYỆN (Sanity Check trên `data/final/` và DataLoader)

**Mục tiêu:** Thực hiện "Sanity Check" để đảm bảo các đoạn code Pipeline (Format, Landmark Extraction, Split, Balance, Custom DataLoader) hoạt động hoàn hảo và mô hình sẽ "nhìn thấy" dữ liệu đúng như kỳ vọng.

### 2.1 Kiểm định Tỷ lệ Phân chia (Split Ratio Validation)

-   **Thao tác:** Đếm tổng số lượng file trong 3 thư mục `train`, `val`, `test` dưới đường dẫn `data/final/`.
    
-   **Cấu trúc thư mục kiểm định:**
    ```
    data/final/
    ├── train/
    │   ├── images/
    │   └── graphs/
    ├── val/
    │   ├── images/
    │   └── graphs/
    └── test/
        ├── images/
        └── graphs/
    ```
-   **Insight:** Xác nhận số lượng mẫu độc lập (không tính phần oversampling) tuân thủ tỷ lệ phân chia ban đầu `70% - 15% - 15%` sử dụng kỹ thuật Stratified Split.

### 2.2 Kiểm định Tính công bằng / Cân bằng nhãn (Label Balancing Check)

-   **Thao tác:** Đếm số lượng tệp tin ảnh (`.jpg`/`.png`) và tệp tin đồ thị (`.npy`) của 4 nhãn trong cả 3 tập (Train, Val, Test).
    
-   **Trực quan hóa:** Biểu đồ cột (Bar Chart) số lượng mẫu của từng nhãn trong 3 tập.
    
-   **Insight (RẤT QUAN TRỌNG):**
    -   **Biểu đồ Train:** Cả 4 nhãn phải bằng nhau tuyệt đối ở mốc **4,500 mẫu**.
    -   **Sự đồng bộ:** Số lượng tệp ảnh trong `train/images/{label}` và số lượng tệp đồ thị trong `train/graphs/{label}` phải trùng khớp 100% đối với mọi nhãn.
    -   **Biểu đồ Val & Test:** 4 cột nhãn giữ nguyên độ nhấp nhô theo phân bố tự nhiên (nhằm phản ánh đúng hiệu năng thực tế của mô hình và tránh rò rỉ dữ liệu).

### 2.3 Kiểm định Shape & Format Kỹ thuật

-   **Thao tác:** Quét ngẫu nhiên một số lượng mẫu nhất định trong thư mục dữ liệu cuối cùng và đọc bằng numpy/OpenCV.
    
-   **Tiêu chuẩn kiểm tra:**
    -   100% ảnh đọc từ `images/` phải có kích thước chuẩn `(224, 224, 1)` (ảnh xám) và giá trị pixel nằm trong đoạn `[0, 255]` trước khi nạp vào DataLoader.
    -   100% đồ thị đọc từ `graphs/` (định dạng `.npy`) phải có shape `(468, 2)` (tọa độ $(x,y)$ của 468 landmark khuôn mặt).

### 2.4 Trực quan hóa Lăng kính Mô hình (DataLoader & Augmentation Sanity Check)

-   **Thao tác:** Import hàm `get_data_loader()` từ file [data_loader.py](file:///Users/anhnn/Documents/lectures/mentor-fptu/support/sem-09/dpl302m-fer-dual/src/data/data_loader.py). Khởi tạo `train_ds` với các cấu hình mong muốn (ví dụ: `to_rgb=True` để phù hợp với Transfer Learning, `augment=True` để bật augmentation độ sáng). Rút ra 1 batch đầu tiên.
    
-   **Đánh giá cấu trúc đầu ra:**
    -   Kiểm tra xem batch trả về có đúng định dạng tuple: `((img_batch, graph_batch), label_batch)` hay không.
    -   Độ lớn của chiều dữ liệu:
        -   `img_batch.shape` phải là `(batch_size, 224, 224, 3)` (nếu `to_rgb=True`) hoặc `(batch_size, 224, 224, 1)` (nếu `to_rgb=False`).
        -   `graph_batch.shape` phải là `(batch_size, 468, 2)`.
        -   `label_batch.shape` phải là `(batch_size, 4)`.
    -   Trực quan hóa bằng đồ họa: Hiển thị lưới ảnh của một batch lên màn hình. Đảm bảo ảnh đã được chuẩn hóa về đoạn `[0.0, 1.0]` và nhận diện rõ hiệu ứng của Data Augmentation (như thay đổi độ sáng ngẫu nhiên).
