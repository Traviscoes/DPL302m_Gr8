# KIẾN TRÚC HỆ THỐNG VÀ QUY TRÌNH THỰC THI (END-TO-END PIPELINE)

Quy trình phát triển hệ thống được chia thành 4 giai đoạn độc lập nhưng liên kết chặt chẽ nhằm đảm bảo tính toàn vẹn dữ liệu và tính khả thi khi triển khai mô hình.

---

## Giai đoạn 1: Data Pipeline (Luồng tiền xử lý & Đồng bộ hóa)

Đây là tập hợp các script chạy offline (một lần) để xây dựng bộ dữ liệu đồng bộ chuẩn trước khi huấn luyện:

1.  **Ingestion & Filtration:** Đọc dữ liệu từ 3 bộ dữ liệu gốc (RAF-DB, FER2013, CK+), lọc và đưa về cấu trúc thư mục đồng nhất với 4 cảm xúc mục tiêu (`anger`, `happiness`, `neutral`, `sadness`) trong thư mục `data/preprocessed/`.
2.  **Format Images:** Định dạng lại toàn bộ ảnh về ảnh xám (Grayscale - 1 kênh), kích thước `224x224` và lưu vào thư mục `data/processed/images/`.
3.  **Extract Landmarks (Graphs):** Dùng mô hình `MediaPipe Tasks API` chạy trên các ảnh đã định dạng để trích xuất tọa độ $(x, y)$ của 468 điểm mốc khuôn mặt. Kết quả lưu thành ma trận `.npy` kích thước `(468, 2)` trong thư mục `data/processed/graphs/`.
    -   *Luật đồng bộ:* Bất kỳ ảnh nào không trích xuất được landmarks sẽ bị xóa khỏi thư mục `images/` để đảm bảo tính đồng bộ 1-1 tuyệt đối giữa ảnh và đồ thị landmark.
4.  **Split & Balance:** Phân chia tập dữ liệu theo tỷ lệ ngẫu nhiên phân tầng `70% Train - 15% Val - 15% Test`. Tiến hành cân bằng tập **Train** (undersampling/oversampling) để đạt chính xác **4,500 mẫu mỗi lớp**. Việc cân bằng được áp dụng song hành cho cả ảnh và file đồ thị tương ứng. Dữ liệu cuối cùng được ghi vào thư mục `data/final/`.

---

## Giai đoạn 2: Training Pipeline (Luồng huấn luyện mô hình)

Luồng huấn luyện được chạy thông qua Jupyter Notebook (hoặc script huấn luyện) sử dụng môi trường GPU:

1.  **Data Loading (Custom Dataset Loader):** Thay thế Keras `ImageDataGenerator` bằng Custom Dataset dựng trên `tf.data.Dataset`. Hàm loader sẽ đọc song song tệp ảnh (`.jpg`/`.png`) và tệp landmark (`.npy`), trả về tuple dữ liệu chuẩn: `((Image_Tensor, Graph_Tensor), Label_OneHot)`.
    -   *Hỗ trợ tương thích:* Loader hỗ trợ nhân bản kênh màu Grayscale thành 3 kênh màu (`to_rgb=True`) trên RAM để tương thích với các mạng pre-trained (như MobileNetV2, ResNet50) mà không tốn tài nguyên ổ cứng.
2.  **Data Augmentation:** Áp dụng tăng cường dữ liệu an toàn (như thay đổi độ sáng ngẫu nhiên - Random Brightness) trực tiếp trong loader. Tránh áp dụng các phép biến đổi hình học (xoay, lật) nhằm bảo toàn tính đồng bộ tọa độ landmark.
3.  **Model Compile & Fit:** Thiết lập kiến trúc mạng **Dual-Branch GCN-CNN**. Biên dịch mô hình với hàm loss `CategoricalCrossentropy` và thuật toán tối ưu `Adam`. Huấn luyện mô hình và sử dụng các Callbacks tự động (Early Stopping, Learning Rate Scheduler, Model Checkpoint).
4.  **Model Export:** Lưu mô hình có `val_accuracy` cao nhất dưới định dạng `.keras` phục vụ cho việc đánh giá và triển khai POC.

---

## Giai đoạn 3: Evaluation & Analysis Pipeline (Đánh giá chéo)

1.  Nạp mô hình `.keras` tốt nhất đã được huấn luyện.
2.  Thực hiện dự đoán (Predict) trên tập Testing (giữ nguyên phân phối tự nhiên để đảm bảo tính khách quan).
3.  Trích xuất chỉ số *Precision, Recall, F1-Score* và xuất tệp báo cáo chi tiết (`classification_report.txt`).
4.  Vẽ và lưu trữ biểu đồ lịch sử huấn luyện (Loss/Accuracy) cùng ma trận nhầm lẫn chuẩn hóa (Normalized Confusion Matrix) phục vụ báo cáo khoa học.

---

## Giai đoạn 4: Deployment Pipeline (Luồng thực thi POC Real-time)

Sản phẩm ứng dụng thử nghiệm giao diện thực tế sử dụng Webcam cục bộ thông qua OpenCV kết hợp Streamlit/Gradio hoặc giao diện đồ họa trực tiếp:

1.  **Input Video Stream:** Đọc luồng video thời gian thực từ camera (`cv2.VideoCapture(0)`).
2.  **Real-time Pre-processing:** Đối với mỗi frame hình ảnh:
    -   Chạy thuật toán phát hiện và cắt (crop) khuôn mặt xuất hiện trong khung hình.
    -   Định dạng khuôn mặt đã cắt về ảnh xám (Grayscale), kích thước `224x224` pixels.
    -   Đưa khuôn mặt qua mô hình `MediaPipe Tasks API` để trích xuất ma trận landmark `(468, 2)`.
    -   Chuẩn hóa ảnh xám về khoảng `[0.0, 1.0]` (và nhân bản kênh lên `3` kênh nếu mô hình yêu cầu RGB).
3.  **Inference (Dự đoán song song):** Đưa đồng thời tensor ảnh khuôn mặt và ma trận landmark vào mô hình Dual-Branch để lấy phân phối xác suất của 4 cảm xúc.
4.  **Overlay & Display:**
    -   Lấy ra nhãn có xác suất cao nhất.
    -   Vẽ khung hình chữ nhật (Bounding Box) quanh khuôn mặt trên khung hình gốc.
    -   Hiển thị văn bản (Text) biểu thị cảm xúc và độ tin cậy tương ứng (VD: "Neutral: 91%") ngay phía trên bounding box.
    -   Hiển thị luồng video đã chèn thông tin ra màn hình người dùng với tốc độ phản hồi mượt mà (FPS tối ưu).
