"""
Script: src/data/format_data.py
Description: 
    - Đọc ảnh trực tiếp từ tập preprocessed (CK+, FER-2013, RAF-DB).
    - Chuẩn hóa toàn bộ về ảnh XÁM (Grayscale - 1 kênh màu).
    - Resize toàn bộ ảnh về kích thước chuẩn 224x224.
    - Lưu vào thư mục processed/images.
"""

import cv2
import logging
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple

# =============================================================================
# CẤU HÌNH (CONFIGURATION)
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Định nghĩa các đường dẫn
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"

# Định nghĩa nhánh đầu ra (Chỉ giữ nhánh ảnh 224x224)
OUTPUT_IMAGES_DIR = PROJECT_ROOT / "data" / "processed" / "images"

# Kích thước ảnh chuẩn (224x224)
SIZE_IMAGE = (224, 224)

# Chuẩn hóa nhãn (Enum Map)
LABEL_MAPPING = {
    # Các thư mục đã đúng chuẩn (CK+, RAF-DB và nhãn neutral của FER-2013)
    'anger': 'anger',
    'happiness': 'happiness',
    'neutral': 'neutral',
    'sadness': 'sadness',
    
    # Các thư mục cần ánh xạ đổi tên của riêng FER-2013
    'angry': 'anger',
    'happy': 'happiness',
    'sad': 'sadness',
}

STANDARD_LABELS = ['anger', 'happiness', 'neutral', 'sadness']

# =============================================================================
# HÀM XỬ LÝ LÕI (CORE FUNCTIONS)
# =============================================================================

def setup_directories(output_dir: Path):
    """Tạo sẵn cây thư mục đích cho nhánh ảnh."""
    if output_dir.exists():
        logger.warning(f"Thư mục {output_dir.relative_to(PROJECT_ROOT)} đã tồn tại. Dữ liệu mới sẽ được ghi đè/bổ sung.")
    for label in STANDARD_LABELS:
        (output_dir / label).mkdir(parents=True, exist_ok=True)


def scan_datasets(input_dir: Path) -> List[Tuple[Path, str, str]]:
    """
    Duyệt toàn bộ thư mục preprocessed để gom đường dẫn ảnh.
    Returns: List [(Đường_dẫn_ảnh, Nhãn_chuẩn, Tên_dataset)]
    """
    logger.info(f"Đang quét dữ liệu tại: {input_dir.relative_to(PROJECT_ROOT)}")
    image_records = []
    
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        for img_path in input_dir.rglob(ext):
            parent_folder_name = img_path.parent.name.lower()
            
            if parent_folder_name in LABEL_MAPPING:
                standard_label = LABEL_MAPPING[parent_folder_name]
                
                # Trích xuất tên dataset gốc để đặt prefix, tránh trùng tên file
                try:
                    relative_parts = img_path.relative_to(input_dir).parts
                    dataset_source = relative_parts[0]
                except ValueError:
                    dataset_source = "unknown"
                    
                image_records.append((img_path, standard_label, dataset_source))
                
    logger.info(f"Đã tìm thấy {len(image_records)} ảnh hợp lệ.")
    return image_records


def process_and_save_image(img_path: Path, standard_label: str, dataset_source: str, 
                           output_dir: Path) -> bool:
    """
    Đọc ảnh, chuyển Grayscale, resize về 224x224 và lưu vào thư mục output_dir.
    Trả về True nếu thành công, False nếu lỗi.
    """
    # 1. Đọc ảnh trực tiếp ở chế độ Grayscale (1 kênh màu)
    img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    
    if img_gray is None:
        return False
        
    # 2. Tạo tên file mới có prefix
    new_filename = f"{dataset_source}_{img_path.name}"
    
    # 3. Xử lý resize về 224x224
    # Dùng INTER_CUBIC (hoặc LINEAR) vì ảnh gốc (đặc biệt là FER-2013 48x48) sẽ bị phóng to lên
    img_resized = cv2.resize(img_gray, SIZE_IMAGE, interpolation=cv2.INTER_CUBIC)
    path_dest = output_dir / standard_label / new_filename
    cv2.imwrite(str(path_dest), img_resized)
    
    return True

# =============================================================================
# LUỒNG THỰC THI CHÍNH
# =============================================================================

def main():
    logger.info("BẮT ĐẦU ĐỊNH DẠNG DỮ LIỆU (DATA FORMATTING)")
    logger.info(f" -> Nhánh Ảnh (Images): Grayscale {SIZE_IMAGE}")
    
    # Khởi tạo thư mục đích
    setup_directories(OUTPUT_IMAGES_DIR)
    
    # Quét dữ liệu đầu vào
    image_records = scan_datasets(PREPROCESSED_DIR)
    if not image_records:
        logger.error("Không tìm thấy dữ liệu. Hãy kiểm tra lại data/preprocessed/")
        return

    # Thống kê
    stats = {'processed': 0, 'success': 0, 'errors': 0}

    # Vòng lặp chính với thanh tiến trình
    for img_path, standard_label, dataset_source in tqdm(image_records, desc="Đang xử lý ảnh", unit="img"):
        stats['processed'] += 1
        
        success = process_and_save_image(
            img_path=img_path,
            standard_label=standard_label,
            dataset_source=dataset_source,
            output_dir=OUTPUT_IMAGES_DIR
        )
        
        if success:
            stats['success'] += 1
        else:
            stats['errors'] += 1

    # Báo cáo
    logger.info("="*50)
    logger.info("HOÀN TẤT ĐỊNH DẠNG DỮ LIỆU!")
    logger.info(f"Tổng số ảnh quét   : {stats['processed']}")
    logger.info(f"Thành công         : {stats['success']} (Đã lưu vào processed/images/)")
    logger.info(f"Lỗi đọc file       : {stats['errors']}")
    logger.info("="*50)


if __name__ == "__main__":
    main()
