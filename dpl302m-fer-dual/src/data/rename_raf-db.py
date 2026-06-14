"""
Script: src/data/rename_rafdb.py
Description: 
    - Đổi tên các thư mục nhãn dạng số (4, 5, 6, 7) của bộ dữ liệu RAF-DB 
      thành định dạng chữ chuẩn (happiness, sadness, anger, neutral).
    - Thao tác trực tiếp trên thư mục data/preprocessed/raf-db/DATASET
"""

import logging
import shutil
from pathlib import Path

# =============================================================================
# CẤU HÌNH (CONFIGURATION)
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Xác định đường dẫn gốc của project
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Đường dẫn đến thư mục chứa ảnh RAF-DB đã qua bước lọc
# Dựa vào script filter_rafdb.py trước đó, cấu trúc đang là: preprocessed/raf-db/DATASET/train và test
RAFDB_ROOT = PROJECT_ROOT / "data" / "preprocessed" / "raf-db"
RAFDB_DATASET_DIR = RAFDB_ROOT / "DATASET"

# Từ điển Mapping đổi tên
RENAME_MAP = {
    '4': 'happiness',
    '5': 'sadness',
    '6': 'anger',
    '7': 'neutral'
}

# =============================================================================
# HÀM XỬ LÝ (CORE FUNCTIONS)
# =============================================================================

def rename_folders_in_split(split_name: str, base_dir: Path):
    """
    Hàm thực hiện đổi tên thư mục cho một tệp con (train hoặc test).
    """
    split_dir = base_dir / split_name
    
    if not split_dir.exists():
        logger.error(f"Không tìm thấy thư mục: {split_dir}")
        return

    logger.info(f"Đang xử lý tập [{split_name.upper()}] tại: {split_dir}")
    
    # Lặp qua các nhãn cần đổi tên
    for old_name, new_name in RENAME_MAP.items():
        old_path = split_dir / old_name
        new_path = split_dir / new_name
        
        # Kiểm tra nếu thư mục cũ (số) tồn tại thì mới tiến hành đổi tên
        if old_path.exists():
            # Nếu vì lý do nào đó thư mục mới đã tồn tại, ta sẽ cảnh báo
            if new_path.exists():
                logger.warning(f"Thư mục đích '{new_name}' đã tồn tại! Cần kiểm tra thủ công.")
            else:
                # Thực hiện đổi tên
                old_path.rename(new_path)
                logger.info(f"Đã đổi tên: '{old_name}' ➔ '{new_name}'")
        else:
            # Nếu thư mục cũ không tồn tại, có thể nó đã được đổi tên từ lần chạy trước
            if new_path.exists():
                logger.info(f"Thư mục '{new_name}' đã tồn tại sẵn (có thể đã chạy script này rồi).")
            else:
                logger.warning(f"Không tìm thấy thư mục gốc '{old_name}/' để đổi tên.")

def cleanup_and_move_directories(current_base_dir: Path):
    """
    Di chuyển thư mục train/, test/ ra ngoài raf-db/ (nếu cần)
    Sau đó xóa folder DATASET/ và các file labels CSV không còn cần thiết.
    """
    logger.info("BẮT ĐẦU DỌN DẸP VÀ TỔ CHỨC LẠI THƯ MỤC...")
    
    # 1. Di chuyển train và test ra folder raf-db (nếu đang ở trong DATASET/)
    if current_base_dir != RAFDB_ROOT:
        for split in ['train', 'test']:
            src_dir = current_base_dir / split
            dst_dir = RAFDB_ROOT / split
            
            if src_dir.exists():
                # Nếu thư mục đích đã tồn tại, xóa đi trước khi move
                if dst_dir.exists():
                    shutil.rmtree(dst_dir)
                shutil.move(str(src_dir), str(dst_dir))
                logger.info(f"Đã di chuyển: {split}/ ➔ {dst_dir.relative_to(PROJECT_ROOT)}")
                
        # 2. Xóa folder DATASET (chỉ xóa nếu nó tồn tại)
        if current_base_dir.exists():
            shutil.rmtree(current_base_dir)
            logger.info(f"Đã xóa thư mục rỗng: {current_base_dir.name}/")
    else:
        logger.info("Thư mục train/ và test/ đã ở đúng vị trí gốc, bỏ qua bước di chuyển.")

    # 3. Xóa các file CSV
    for csv_file in ['train_labels.csv', 'test_labels.csv']:
        csv_path = RAFDB_ROOT / csv_file
        if csv_path.exists():
            csv_path.unlink() # Lệnh unlink() của thư viện Path tương đương với os.remove()
            logger.info(f"Đã xóa file: {csv_file}")


# =============================================================================
# LUỒNG THỰC THI CHÍNH
# =============================================================================

def main():
    logger.info("BẮT ĐẦU CHUẨN HÓA TÊN NHÃN CHO RAF-DB")
    
    # Sử dụng biến cục bộ để lưu vị trí hiện tại của dữ liệu
    current_base_dir = RAFDB_DATASET_DIR
    
    # Kiểm tra xem dataset đang ở trạng thái nào
    if not current_base_dir.exists():
        if (RAFDB_ROOT / "train").exists() and (RAFDB_ROOT / "test").exists():
            logger.info("Thư mục train/ và test/ đã được đưa ra ngoài. Sẽ tiến hành kiểm tra đổi tên ở vị trí mới.")
            # Gán lại giá trị cho biến cục bộ thay vì dùng global
            current_base_dir = RAFDB_ROOT
        else:
            logger.error(f"Đường dẫn dataset không hợp lệ: {RAFDB_ROOT}")
            logger.info("Hãy chắc chắn bạn đã chạy file src/data/filter_rafdb.py trước.")
            return

    # 1. Đổi tên các folder số thành tên cảm xúc
    for split in ['train', 'test']:
        rename_folders_in_split(split, current_base_dir)

    # 2. Dọn dẹp và tổ chức lại folder
    cleanup_and_move_directories(current_base_dir)

    logger.info("="*50)
    logger.info("HOÀN TẤT ĐỔI TÊN VÀ DỌN DẸP! Cấu trúc RAF-DB đã sẵn sàng.")
    logger.info("="*50)


if __name__ == "__main__":
    main()
