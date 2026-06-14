"""
Script: src/data/split_balance.py
Description: 
    - Đọc dữ liệu đã format (từ data/processed/images/ và data/processed/graphs/).
    - Thực hiện Stratified Split: 70% Train - 15% Val - 15% Test.
    - Áp dụng Data Balancing (Undersampling & Oversampling) ĐỘC QUYỀN trên tập Train.
    - Đồng bộ hóa hoàn hảo giữa ảnh (.jpg/.png) và landmark (.npy) khi nhân bản/loại bỏ.
    - Lưu output ra: data/final/
"""

import logging
import shutil
import random
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# =============================================================================
# CẤU HÌNH HỆ THỐNG
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Thiết lập đường dẫn
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FINAL_DIR = PROJECT_ROOT / "data" / "final"

LABELS = ['anger', 'happiness', 'neutral', 'sadness']

# Cấu hình siêu tham số (Hyperparameters)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42 # Cố định seed để có thể tái lập kết quả

# Mục tiêu số lượng ảnh mỗi class trong tập Train
TARGET_SAMPLES_PER_CLASS = 4500 

# =============================================================================
# HÀM XỬ LÝ LÕI (CORE LOGIC)
# =============================================================================

def get_base_inventory() -> Tuple[List[str], List[str]]:
    """
    Quét thư mục processed/images để lấy danh sách toàn bộ file ảnh và nhãn tương ứng.
    Returns: (X_filenames, y_labels)
    """
    images_dir = PROCESSED_DIR / 'images'
    if not images_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {images_dir}")

    X_filenames = []
    y_labels = []

    for label in LABELS:
        label_dir = images_dir / label
        if not label_dir.exists():
            continue
            
        for img_path in label_dir.glob('*.*'):
            if img_path.is_file() and not img_path.name.startswith('.'):
                X_filenames.append(img_path.name)
                y_labels.append(label)

    return X_filenames, y_labels


def perform_stratified_split(X: List[str], y: List[str]) -> Dict[str, Dict[str, List[str]]]:
    """
    Chia 70-15-15 có phân tầng (Stratified) dựa trên nhãn y.
    Returns: Dictionary chứa danh sách file theo cấu trúc: {split: {label: [files]}}
    """
    logger.info("Thực hiện Stratified Split (70% Train, 15% Val, 15% Test)...")
    
    # 1. Tách Test set trước (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, stratify=y, random_state=RANDOM_SEED
    )
    
    # 2. Tách Val set từ phần còn lại (Val chiếm 15% của TỔNG, tức là 15/85 của phần còn lại)
    val_relative_ratio = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative_ratio, stratify=y_temp, random_state=RANDOM_SEED
    )

    # Tổ chức lại data thành dạng Dictionary để dễ duyệt
    def group_by_label(files, labels):
        grouped = {l: [] for l in LABELS}
        for f, l in zip(files, labels):
            grouped[l].append(f)
        return grouped

    return {
        'train': group_by_label(X_train, y_train),
        'val': group_by_label(X_val, y_val),
        'test': group_by_label(X_test, y_test)
    }


def balance_training_set(train_dict: Dict[str, List[str]]) -> Dict[str, List[Tuple[str, str]]]:
    """
    Áp dụng Undersampling & Oversampling cho riêng tập Train.
    Returns: Dict chứa danh sách Tuple (Tên_file_gốc, Tên_file_đích_sau_khi_copy)
    """
    logger.info(f"Cân bằng tập Train về mốc {TARGET_SAMPLES_PER_CLASS} ảnh/lớp...")
    balanced_train = {}

    for label, files in train_dict.items():
        current_count = len(files)
        balanced_files = []
        
        if current_count == 0:
            logger.warning(f"Class '{label}' không có ảnh nào trong tập Train!")
            balanced_train[label] = []
            continue

        if current_count > TARGET_SAMPLES_PER_CLASS:
            # UNDERSAMPLING: Chọn ngẫu nhiên N ảnh
            logger.info(f"  - {label.upper()}: Undersampling ({current_count} -> {TARGET_SAMPLES_PER_CLASS})")
            selected_files = random.sample(files, TARGET_SAMPLES_PER_CLASS)
            # Tên gốc và tên đích giữ nguyên
            balanced_files = [(f, f) for f in selected_files]
            
        elif current_count < TARGET_SAMPLES_PER_CLASS:
            # OVERSAMPLING: Lấy toàn bộ ảnh gốc, cộng thêm ảnh chọn ngẫu nhiên có hoàn lại (with replacement)
            logger.info(f"  - {label.upper()}: Oversampling ({current_count} -> {TARGET_SAMPLES_PER_CLASS})")
            
            # 1. Thêm toàn bộ file gốc
            balanced_files.extend([(f, f) for f in files])
            
            # 2. Tính số lượng cần bù đắp và nhân bản ngẫu nhiên
            deficit = TARGET_SAMPLES_PER_CLASS - current_count
            oversampled_files = random.choices(files, k=deficit)
            
            # Đổi tên file nhân bản để tránh ghi đè (ví dụ: img_1.jpg -> img_1_copy_0.jpg)
            for i, f in enumerate(oversampled_files):
                base_name = f.rsplit('.', 1)[0]
                ext = f.rsplit('.', 1)[1]
                new_name = f"{base_name}_copy_{i}.{ext}"
                balanced_files.append((f, new_name))
        else:
            logger.info(f"  - {label.upper()}: Đã đạt chuẩn ({current_count})")
            balanced_files = [(f, f) for f in files]

        random.shuffle(balanced_files) # Trộn đều để tránh các file copy nằm liền nhau
        balanced_train[label] = balanced_files

    return balanced_train


# =============================================================================
# FILE SYSTEM I/O (WORKER)
# =============================================================================

def setup_final_directories():
    """Tạo mới/Làm sạch cấu trúc thư mục đích trong data/final."""
    if FINAL_DIR.exists():
        logger.warning("Thư mục data/final đã tồn tại. Tiến hành xóa để tạo lại cấu trúc sạch...")
        shutil.rmtree(FINAL_DIR)
        
    for split in ['train', 'val', 'test']:
        for data_type in ['images', 'graphs']:
            for label in LABELS:
                (FINAL_DIR / split / data_type / label).mkdir(parents=True, exist_ok=True)


def copy_files(split_map: Dict, balanced_train_map: Dict):
    """
    Thực thi việc copy file vật lý từ processed/ sang final/ đồng bộ cho cả images và graphs.
    """
    logger.info("Bắt đầu di chuyển dữ liệu vật lý (Copying files)...")

    # 1. Tập TRAIN (Sử dụng balanced_train_map để cân bằng lớp)
    for label, file_tuples in tqdm(balanced_train_map.items(), desc="Train (Images & Graphs)"):
        src_img_dir = PROCESSED_DIR / 'images' / label
        src_graph_dir = PROCESSED_DIR / 'graphs' / label
        
        dst_img_dir = FINAL_DIR / 'train' / 'images' / label
        dst_graph_dir = FINAL_DIR / 'train' / 'graphs' / label

        for original_name, dest_name in file_tuples:
            # Sao chép ảnh
            shutil.copy2(src_img_dir / original_name, dst_img_dir / dest_name)
            
            # Đồng bộ sao chép landmark (.npy)
            original_npy = Path(original_name).with_suffix('.npy').name
            dest_npy = Path(dest_name).with_suffix('.npy').name
            shutil.copy2(src_graph_dir / original_npy, dst_graph_dir / dest_npy)

    # 2. Tập VAL và TEST (Sử dụng split_map gốc, không bị biến đổi)
    for split in ['val', 'test']:
        for label, files in tqdm(split_map[split].items(), desc=f"{split.capitalize()} (Images & Graphs)"):
            src_img_dir = PROCESSED_DIR / 'images' / label
            src_graph_dir = PROCESSED_DIR / 'graphs' / label
            
            dst_img_dir = FINAL_DIR / split / 'images' / label
            dst_graph_dir = FINAL_DIR / split / 'graphs' / label
            
            for f in files:
                # Sao chép ảnh
                shutil.copy2(src_img_dir / f, dst_img_dir / f)
                
                # Đồng bộ sao chép landmark (.npy)
                npy_file = Path(f).with_suffix('.npy').name
                shutil.copy2(src_graph_dir / npy_file, dst_graph_dir / npy_file)


# =============================================================================
# LUỒNG THỰC THI CHÍNH
# =============================================================================

def main():
    logger.info("="*60)
    logger.info("PHASE 3: SPLIT & BALANCE DATASET")
    logger.info("="*60)

    try:
        # Bước 1: Lấy danh sách toàn bộ ảnh
        X_all, y_all = get_base_inventory()
        logger.info(f"Tổng số ảnh thu thập được: {len(X_all)}")
        
        # Bước 2: Tính toán bản đồ chia tập (Stratified)
        split_map = perform_stratified_split(X_all, y_all)
        
        # Bước 3: Tính toán cân bằng riêng cho tập Train
        balanced_train_map = balance_training_set(split_map['train'])
        
        # Bước 4: Tạo cấu trúc thư mục sạch
        setup_final_directories()
        
        # Bước 5: Thực thi I/O (Copy file đồng bộ)
        copy_files(split_map, balanced_train_map)

        logger.info("="*60)
        logger.info("HOÀN TẤT! Dữ liệu đã sẵn sàng tại: data/final/")
        logger.info("="*60)

    except Exception as e:
        logger.exception(f"Quá trình thất bại: {e}")


if __name__ == "__main__":
    # Cài đặt seed cho module random để Over/Under-sampling luôn ra kết quả giống nhau ở các lần chạy
    random.seed(RANDOM_SEED)
    main()
