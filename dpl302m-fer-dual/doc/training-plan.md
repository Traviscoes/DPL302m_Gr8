# CHI TIẾT CHIẾN LƯỢC VÀ ĐÁNH GIÁ MÔ HÌNH (MODELING STRATEGY)

Dự án sẽ thực nghiệm kiến trúc mô hình học sâu kết hợp giữa dữ liệu cấu trúc bề mặt khuôn mặt (Texture) và dữ liệu hình học liên kết (Geometry) thông qua mạng hai nhánh **Dual-Branch GCN-CNN**. Để có cơ sở đối chiếu khoa học, dự án xây dựng một mô hình baseline thuần hình ảnh trước khi huấn luyện mô hình tích hợp hai nhánh.

Tất cả các mô hình đều được huấn luyện trên luồng dữ liệu chuẩn hóa từ module [data_loader.py](file:///Users/anhnn/Documents/lectures/mentor-fptu/support/sem-09/dpl302m-fer-dual/src/data/data_loader.py) với 4 nhãn cảm xúc: `anger, happiness, neutral, sadness`.

---

## 1. Mô hình Baseline: MobileNetV2 (Image-Only Branch)

-   **Mục tiêu:** Thiết lập mốc hiệu năng chuẩn (baseline benchmark) thuần hình ảnh sử dụng phương pháp Transfer Learning, làm cơ sở so sánh với mô hình tích hợp đồ thị landmark.
-   **Đầu vào (Input):** Ảnh xám kích thước `(224, 224, 1)` được Custom DataLoader nhân bản tự động lên 3 kênh `(224, 224, 3)` trên bộ nhớ RAM để tương thích với cấu trúc mạng MobileNetV2 pre-trained trên tập ImageNet.
-   **Kiến trúc chi tiết:**
    -   **Backbone:** `MobileNetV2` (load trọng số `imagenet`, đóng băng toàn bộ phần xương sống).
    -   **Classifier:** Đầu ra của backbone được đưa qua lớp `GlobalAveragePooling2D` -> Lớp `Dense(128, ReLU)` -> Lớp `Dropout(0.3)` chống overfitting -> Lớp phân loại đầu ra `Dense(4, Softmax)` ứng với 4 cảm xúc.
-   **Ưu điểm:** Khởi động và hội tụ nhanh, lượng tham số huấn luyện ít (chỉ tập trung ở các lớp Dense trên cùng), tận dụng tốt các đặc trưng biên và hình ảnh chất lượng cao từ ImageNet.
-   **Kịch bản chạy:** Huấn luyện trong **10 epochs**, lưu tệp mô hình thành `baseline_mobilenetv2.keras`, xuất biểu đồ học tập và ma trận nhầm lẫn để làm benchmark đối chiếu trực tiếp.

---

## 2. Mô hình Đề xuất: Dual-Branch GCN-CNN Architecture

Mô hình mục tiêu của dự án tích hợp đồng thời hai nguồn thông tin độc lập từ khuôn mặt:

### Nhánh 1: CNN Branch (Trích xuất đặc trưng bề mặt - Texture)
-   **Đầu vào:** Ảnh khuôn mặt kích thước `(224, 224, 1)` (hoặc 3 kênh màu).
-   **Cấu trúc:** Sử dụng kiến trúc CNN (có thể là một mạng Custom CNN gọn nhẹ hoặc MobileNetV2/ResNet đóng vai trò feature extractor).
-   **Đầu ra:** Vector đặc trưng bề mặt (ví dụ kích thước 128 hoặc 256 chiều).

### Nhánh 2: GCN Branch (Trích xuất đặc trưng hình học - Geometry)
-   **Đầu vào:** Ma trận landmark tọa độ $(x, y)$ kích thước `(468, 2)`.
-   **Kiến trúc:** Mạng Tích chập Đồ thị (Graph Convolutional Network - GCN). Định nghĩa khuôn mặt như một đồ thị với các đỉnh (nodes) là các landmark và các cạnh (edges) kết nối dựa trên cấu trúc sinh học khuôn mặt của Face Mesh.
-   **Đầu ra:** Vector đặc trưng cấu trúc hình học (ví dụ kích thước 64 hoặc 128 chiều).

### Khâu kết hợp đặc trưng (Feature Fusion) & Phân loại
-   **Fusion:** Hai vector đặc trưng từ nhánh CNN và nhánh GCN được nối lại với nhau (Concatenate) thành một vector đặc trưng hợp nhất chứa đựng đầy đủ thông tin về nếp nhăn/đường nét bề mặt và khoảng cách biến động hình học giữa các cơ mặt.
-   **Classifier:** Vector hợp nhất đi qua các lớp Fully Connected (Dense) có Dropout và kết thúc bằng lớp phân loại Softmax để đưa ra dự báo xác suất cho 4 nhóm cảm xúc.

---

## 3. Các Siêu tham số (Hyperparameters) dùng chung

Để đảm bảo tính khách quan khi huấn luyện và so sánh mô hình:

-   **Hàm mất mát (Loss Function):** `CategoricalCrossentropy` (áp dụng cho nhãn đã one-hot encoded).
-   **Thuật toán tối ưu (Optimizer):** `Adam` với tốc độ học ban đầu (Learning Rate) là `1e-4`.
-   **Batch Size:** Cấu hình chuẩn ở mức `64`.
-   **Callbacks kiểm soát:**
    -   `EarlyStopping`: Tự động dừng huấn luyện nếu loss trên tập validation không giảm sau một số lượng epoch quy định (tránh overfitting).
    -   `ReduceLROnPlateau`: Tự động giảm learning rate đi một nửa nếu val_loss đi vào vùng bão hòa nhằm giúp mô hình hội tụ tốt hơn vào cực tiểu toàn cục.
    -   `ModelCheckpoint`: Lưu lại phiên bản mô hình tốt nhất dựa trên độ chính xác tập validation (`val_accuracy`).

---

## 4. Chiến lược Đánh giá (Evaluation & Metrics)

Quá trình đánh giá được thực hiện độc lập trên tập Test (`test_ds`), nơi dữ liệu hoàn toàn giữ nguyên tỷ lệ phân bổ tự nhiên (không bị tác động bởi quá trình Oversampling tập Train).

-   **Báo cáo phân loại (Classification Report):** Đo lường chi tiết chỉ số *Precision, Recall, F1-Score* cho từng nhãn cảm xúc cụ thể nhằm phát hiện sớm xem mô hình có bị lệch/thiên vị cho nhãn nào không.
-   **Ma trận nhầm lẫn chuẩn hóa (Normalized Confusion Matrix):** Trực quan hóa bằng heatmap để phân tích sai số (nhầm lẫn giữa các nhãn tương tự nhau như Sadness và Neutral).
-   **Biểu đồ học tập (Learning Curves):** Theo dõi sự chênh lệch giữa loss/accuracy của tập Train và tập Val theo từng epoch để chuẩn đoán hiện tượng Underfitting hoặc Overfitting.
-   **Tốc độ suy luận thực tế (FPS - Frames Per Second):** Đo lường thời gian xử lý trung bình của mô hình trên CPU đối với một khung ảnh webcam thực tế (bao gồm cả bước crop mặt, chạy MediaPipe trích landmark và chạy inference mô hình) nhằm đảm bảo tính khả thi khi chạy POC thời gian thực.
