# HƯỚNG DẪN HUẤN LUYỆN MÔ HÌNH BASELINE TRÊN GOOGLE COLAB

Tài liệu này hướng dẫn chi tiết quy trình chuẩn bị dữ liệu, cấu hình môi trường, huấn luyện mô hình MobileNetV2 Baseline và lưu trữ logs/kết quả khi thực hiện trên Google Colab.

---

## 1. Chuẩn Bị Dữ Liệu và Mã Nguồn (Local)

Để tránh hiện tượng Google Colab đọc trực tiếp hàng ngàn file nhỏ từ Google Drive (gây ra nghẽn I/O rất nặng và làm chậm tốc độ train lên hàng chục lần), chúng ta bắt buộc phải nén bộ dữ liệu trước khi tải lên Drive.

### Bước 1: Nén tập dữ liệu final
Trên máy local (terminal chạy tại thư mục root của dự án), chạy lệnh nén thư mục `data/final/`:
```bash
zip -r final_dataset.zip data/final/
```
Tệp tin `final_dataset.zip` sẽ chứa cấu trúc:
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

### Bước 2: Tải lên Google Drive
Tạo một thư mục trên Google Drive của bạn có tên là `FER_Project`.
Tải các tệp/thư mục sau lên thư mục `FER_Project` trên Drive:
1.  Tệp tin dữ liệu: `final_dataset.zip`
2.  Thư mục mã nguồn: `src/` (tải nguyên thư mục này lên Drive để Colab có thể sử dụng các module như `src.data.data_loader`).

Cấu trúc trên Drive sẽ như sau:
```
My Drive/
└── FER_Project/
    ├── final_dataset.zip
    └── src/
```

---

## 2. Thiết Lập Môi Trường Trên Google Colab

Khi chạy notebook trên Colab, thực hiện theo thứ tự các bước sau để thiết lập môi trường tối ưu:

### Bước 1: Kích hoạt GPU
Vào menu **Runtime** -> **Change runtime type** -> Chọn **T4 GPU** (hoặc GPU bất kỳ có sẵn) để tăng tốc độ huấn luyện.

### Bước 2: Mount Google Drive
Kết nối Google Drive để tải dữ liệu và lưu kết quả:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Bước 3: Copy và giải nén dữ liệu lên ổ đĩa của Colab
Giải nén dữ liệu trực tiếp vào ổ đĩa ảo của Colab (`/content/data/`) để tối ưu tốc độ đọc file (High-speed local SSD):
```python
# Tạo thư mục chứa dữ liệu local
!mkdir -p /content/data

# Copy file zip từ Drive sang Colab local
!cp /content/drive/MyDrive/FER_Project/final_dataset.zip /content/final_dataset.zip

# Giải nén âm thầm (-q)
!unzip -q /content/final_dataset.zip -d /content/

# Xác minh cấu trúc đã giải nén
!ls -la /content/data/final
```

### Bước 4: Nạp mã nguồn dự án (`src/`)
Đưa thư mục mã nguồn `src/` vào môi trường làm việc của Colab và thêm đường dẫn vào `sys.path`:
```python
# Copy thư mục src từ Drive sang Colab local
!cp -r /content/drive/MyDrive/FER_Project/src /content/src

# Thêm đường dẫn vào PYTHONPATH
import sys
if '/content' not in sys.path:
    sys.path.append('/content')
```

---

## 3. Chiến Lược Ghi Logs và Lưu Trữ Kết Quả

Quá trình huấn luyện sử dụng các callback của TensorFlow để ghi nhận và lưu trữ toàn bộ chỉ số tự động. Tất cả các kết quả này sẽ được lưu trực tiếp vào thư mục báo cáo trên Google Drive của bạn (`/content/drive/MyDrive/FER_Project/reports/` và `/content/drive/MyDrive/FER_Project/models/`) để đảm bảo không bị mất khi mất kết nối hoặc tắt runtime.

### 3.1 Cấu hình các thư mục đầu ra trên Drive
```python
import os
from pathlib import Path

# Định nghĩa các thư mục lưu kết quả trên Drive
DRIVE_PROJECT = Path('/content/drive/MyDrive/FER_Project')
DRIVE_REPORTS = DRIVE_PROJECT / 'reports'
DRIVE_MODELS = DRIVE_PROJECT / 'models'

# Tạo thư mục nếu chưa tồn tại
DRIVE_REPORTS.mkdir(parents=True, exist_ok=True)
DRIVE_MODELS.mkdir(parents=True, exist_ok=True)
```

### 3.2 Bộ Loggers và Callbacks Huấn Luyện
Trong phương thức `model.fit()`, chúng ta khai báo các callback sau:

1.  **Ghi logs chi tiết từng Epoch sang file CSV (`tf.keras.callbacks.CSVLogger`)**:
    Lưu trữ các chỉ số `loss`, `accuracy`, `val_loss`, `val_accuracy`, `lr` của từng epoch vào file CSV.
    ```python
    from tensorflow.keras.callbacks import CSVLogger
    csv_log_path = DRIVE_REPORTS / 'baseline_training_log.csv'
    csv_logger = CSVLogger(str(csv_log_path), separator=',', append=False)
    ```

2.  **Lưu trọng số mô hình tốt nhất (`tf.keras.callbacks.ModelCheckpoint`)**:
    Lưu mô hình có độ chính xác trên tập validation (`val_accuracy`) tốt nhất dưới định dạng `.keras`.
    ```python
    from tensorflow.keras.callbacks import ModelCheckpoint
    checkpoint_path = DRIVE_MODELS / 'baseline_mobilenetv2.keras'
    checkpoint = ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    ```

3.  **Tự động dừng sớm (`tf.keras.callbacks.EarlyStopping`)**:
    Nếu `val_loss` không cải thiện sau `10` epochs liên tiếp, quá trình train sẽ tự dừng để tránh Overfitting.
    ```python
    from tensorflow.keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    ```

4.  **Tự động giảm tốc độ học (`tf.keras.callbacks.ReduceLROnPlateau`)**:
    Nếu `val_loss` không cải thiện sau `5` epochs liên tiếp, giảm learning rate đi một nửa để mô hình hội tụ tốt hơn.
    ```python
    from tensorflow.keras.callbacks import ReduceLROnPlateau
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    ```

---

## 4. Quy Trình Đánh Giá và Xuất Báo Cáo

Sau khi kết thúc quá trình huấn luyện:

### 4.1 Vẽ và Lưu Biểu Đồ Lịch Sử Huấn Luyện (Learning Curves)
-   Đọc dữ liệu từ tệp log CSV đã lưu hoặc sử dụng đối tượng `history`.
-   Vẽ biểu đồ Loss và Accuracy (so sánh Train vs Val).
-   Lưu biểu đồ dưới dạng ảnh PNG tại: `reports/baseline_training_curves.png`.

### 4.2 Đánh giá trên tập Testing và Ghi File Báo Cáo Phân Loại (Classification Report)
-   Chạy suy luận (Inference) mô hình tốt nhất trên tập Test để dự báo xác suất.
-   Tính toán các chỉ số Precision, Recall, F1-Score cho cả 4 nhãn (`anger`, `happiness`, `neutral`, `sadness`).
-   Ghi toàn bộ thông tin Classification Report định dạng text vào file: `reports/baseline_classification_report.txt`.

### 4.3 Vẽ và Lưu Ma Trận Nhầm Lẫn Chuẩn Hóa (Normalized Confusion Matrix)
-   Tính toán ma trận nhầm lẫn chuẩn hóa từ nhãn thực tế và nhãn dự đoán.
-   Sử dụng Seaborn Heatmap để trực quan hóa chi tiết.
-   Lưu ma trận nhầm lẫn dưới dạng ảnh PNG tại: `reports/baseline_confusion_matrix.png`.
