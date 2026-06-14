"""
Script: src/data/data_loader.py
Description:
    - Định nghĩa bộ nạp dữ liệu Custom Dataset bằng TensorFlow (tf.data.Dataset).
    - Nạp đồng thời ảnh (224x224, Grayscale hoặc RGB) và đồ thị (.npy, 468x2).
    - Hỗ trợ one-hot encoding cho 4 nhãn cảm xúc.
    - Trả về tuple dữ liệu chuẩn: ((Image_Tensor, Graph_Tensor), Label_OneHot).
"""

import logging
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

# Cấu hình log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_DATA_DIR = PROJECT_ROOT / "data" / "final"

LABELS = ['anger', 'happiness', 'neutral', 'sadness']
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}

def load_data_paths(split: str) -> list:
    """
    Quét thư mục final/{split}/ để lấy danh sách các cặp (path_ảnh, path_graph, label).
    """
    split_dir = FINAL_DATA_DIR / split
    img_dir = split_dir / 'images'
    graph_dir = split_dir / 'graphs'
    
    records = []
    if not split_dir.exists():
        logger.warning(f"Thư mục chia tập {split_dir} không tồn tại.")
        return records

    for label in LABELS:
        label_img_dir = img_dir / label
        if not label_img_dir.exists():
            continue
            
        for img_path in label_img_dir.glob('*.*'):
            if img_path.is_file() and not img_path.name.startswith('.'):
                # Đường dẫn file graph tương ứng
                graph_path = graph_dir / label / f"{img_path.stem}.npy"
                if graph_path.exists():
                    records.append((img_path, graph_path, label))
                else:
                    logger.warning(f"Không tìm thấy file graph tương ứng với ảnh: {img_path}")
                    
    return records

def create_dataset_generator(records: list, to_rgb: bool = True, augment: bool = False):
    """
    Tạo hàm generator để dùng với tf.data.Dataset.from_generator.
    """
    def generator():
        # Sao chép và shuffle danh sách bản ghi nếu đang train
        local_records = list(records)
        if augment:
            # Shuffle ngẫu nhiên mỗi epoch
            np.random.shuffle(local_records)
            
        for img_path, graph_path, label in local_records:
            # 1. Đọc và tiền xử lý ảnh (Grayscale)
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
                
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0 # Chuẩn hóa về [0, 1]
            img = np.expand_dims(img, axis=-1)   # Shape: (224, 224, 1)
            
            # Nếu yêu cầu RGB (Solution B cho các pre-trained models)
            if to_rgb:
                img = np.concatenate([img, img, img], axis=-1) # Shape: (224, 224, 3)
                
            # Áp dụng Augmentation đơn giản không làm thay đổi tọa độ Landmark (nếu bật)
            if augment:
                # Ngẫu nhiên đổi độ sáng (Brightness)
                if np.random.rand() > 0.5:
                    factor = np.random.uniform(0.8, 1.2)
                    img = np.clip(img * factor, 0.0, 1.0)
                    
            # 2. Đọc landmark (.npy)
            graph = np.load(str(graph_path)).astype(np.float32) # Shape: (468, 2)
            
            # 3. Một-hot encode nhãn
            one_hot_label = np.zeros(len(LABELS), dtype=np.float32)
            one_hot_label[LABEL_TO_INDEX[label]] = 1.0
            
            yield (img, graph), one_hot_label
            
    return generator

def get_data_loader(split: str, batch_size: int = 32, to_rgb: bool = True, augment: bool = False, shuffle: bool = True):
    """
    Tạo tf.data.Dataset cho một tập dữ liệu cụ thể (train, val, hoặc test).
    """
    records = load_data_paths(split)
    logger.info(f"Đã nạp {len(records)} cặp mẫu dữ liệu cho tập {split.upper()}")
    
    if len(records) == 0:
        raise FileNotFoundError(f"Không có dữ liệu trong tập {split}")
        
    generator_func = create_dataset_generator(records, to_rgb=to_rgb, augment=augment)
    
    # Định nghĩa cấu trúc đầu ra (Signature)
    img_channels = 3 if to_rgb else 1
    output_signature = (
        (
            tf.TensorSpec(shape=(224, 224, img_channels), dtype=tf.float32, name="image_input"),
            tf.TensorSpec(shape=(468, 2), dtype=tf.float32, name="graph_input")
        ),
        tf.TensorSpec(shape=(len(LABELS),), dtype=tf.float32, name="label")
    )
    
    dataset = tf.data.Dataset.from_generator(
        generator_func,
        output_signature=output_signature
    )
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
        
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

if __name__ == "__main__":
    # Test thử DataLoader
    try:
        logger.info("--- TEST DATALOADER ---")
        train_ds = get_data_loader('train', batch_size=16, to_rgb=True, augment=True)
        
        # Lấy thử 1 batch
        for (img_batch, graph_batch), label_batch in train_ds.take(1):
            logger.info(f"Batch Image Shape: {img_batch.shape}")
            logger.info(f"Batch Graph Shape: {graph_batch.shape}")
            logger.info(f"Batch Label Shape: {label_batch.shape}")
            logger.info(f"Label mẫu: {label_batch[0]}")
            
    except Exception as e:
        logger.exception(f"Lỗi khi test DataLoader: {e}")
