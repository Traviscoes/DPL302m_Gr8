"""
Script: src/models/train_baseline.py
Description:
    - Nạp nhánh ảnh từ Custom DataLoader.
    - Xây dựng mô hình Transfer Learning MobileNetV2 (Keras) cho ảnh 224x224x3.
    - Huấn luyện mô hình (10 epochs) và đánh giá trên tập Test.
    - Vẽ và lưu biểu đồ Loss/Accuracy và Confusion Matrix vào thư mục reports/.
"""

import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from src.data.data_loader import get_data_loader, LABELS

# Cấu hình log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "src" / "models"

# Cấu hình siêu tham số
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-4

def build_mobilenet_baseline():
    """
    Xây dựng mô hình MobileNetV2 làm baseline cho nhánh ảnh.
    """
    logger.info("Đang khởi tạo mô hình MobileNetV2...")
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Đóng băng xương sống (backbone)

    # Thêm các lớp phân loại phía trên
    inputs = tf.keras.Input(shape=(224, 224, 3), name="image_input")
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(len(LABELS), activation='softmax', name="output_layer")(x)

    model = tf.keras.Model(inputs, outputs, name="MobileNetV2_Baseline")
    
    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def plot_and_save_history(history, output_path: Path):
    """Vẽ và lưu biểu đồ Loss và Accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Val')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Val')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    logger.info(f"Đã lưu biểu đồ huấn luyện tại: {output_path}")

def plot_and_save_confusion_matrix(y_true, y_pred, output_path: Path):
    """Vẽ và lưu Confusion Matrix."""
    cm = confusion_matrix(y_true, y_pred)
    # Chuẩn hóa ma trận nhầm lẫn
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_norm, 
        annot=cm, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=LABELS, 
        yticklabels=LABELS,
        cbar=True
    )
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    logger.info(f"Đã lưu ma trận nhầm lẫn tại: {output_path}")

def main():
    # 0. Chuẩn bị thư mục báo cáo
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Nạp dữ liệu
    logger.info("Đang nạp dữ liệu từ data/final/...")
    train_ds = get_data_loader('train', batch_size=BATCH_SIZE, to_rgb=True, augment=True, shuffle=True)
    val_ds = get_data_loader('val', batch_size=BATCH_SIZE, to_rgb=True, augment=False, shuffle=False)
    test_ds = get_data_loader('test', batch_size=BATCH_SIZE, to_rgb=True, augment=False, shuffle=False)
    
    # Ánh xạ để chỉ lấy nhánh ảnh (loại bỏ landmark graph) cho mô hình baseline
    train_ds_img = train_ds.map(lambda inputs, label: (inputs[0], label))
    val_ds_img = val_ds.map(lambda inputs, label: (inputs[0], label))
    test_ds_img = test_ds.map(lambda inputs, label: (inputs[0], label))
    
    # 2. Xây dựng mô hình
    model = build_mobilenet_baseline()
    model.summary(print_fn=logger.info)
    
    # 3. Huấn luyện
    logger.info(f"Bắt đầu huấn luyện mô hình trong {EPOCHS} epochs...")
    history = model.fit(
        train_ds_img,
        epochs=EPOCHS,
        validation_data=val_ds_img
    )
    
    # 4. Lưu biểu đồ lịch sử huấn luyện
    plot_and_save_history(history, REPORTS_DIR / "baseline_training_curves.png")
    
    # 5. Lưu mô hình đã train
    model_save_path = MODELS_DIR / "baseline_mobilenetv2.keras"
    model.save(str(model_save_path))
    logger.info(f"Đã lưu file mô hình tại: {model_save_path}")
    
    # 6. Đánh giá trên tập Test và vẽ Confusion Matrix
    logger.info("Đang đánh giá mô hình trên tập Test...")
    
    y_true_list = []
    y_pred_list = []
    
    # Duyệt thủ công tập Test vì Dataset không shuffle dễ kiểm soát thứ tự
    for img_batch, label_batch in test_ds_img:
        preds = model.predict(img_batch, verbose=0)
        
        y_true_list.extend(np.argmax(label_batch.numpy(), axis=1))
        y_pred_list.extend(np.argmax(preds, axis=1))
        
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    
    # In báo cáo phân loại (Classification Report)
    report = classification_report(y_true, y_pred, target_names=LABELS)
    logger.info("\nClassification Report:\n" + report)
    
    # Lưu Classification Report thành file text
    with open(REPORTS_DIR / "baseline_classification_report.txt", "w") as f:
        f.write(report)
        
    # Vẽ Confusion Matrix
    plot_and_save_confusion_matrix(y_true, y_pred, REPORTS_DIR / "baseline_confusion_matrix.png")
    logger.info("Hoàn tất huấn luyện và đánh giá mô hình baseline!")

if __name__ == "__main__":
    main()
