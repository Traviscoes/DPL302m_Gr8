"""
Script: src/features/extract_landmarks.py
Description:
    - Tải tự động mô hình Face Landmarker của Google (nếu chưa có).
    - Quét toàn bộ ảnh đã định dạng trong data/processed/images.
    - Dùng MediaPipe Tasks API (để tránh lỗi xung đột Protobuf) để trích xuất tọa độ landmark (x, y) của khuôn mặt.
    - Lưu tọa độ dưới dạng file ma trận .npy (shape: 468x2) vào data/processed/graphs.
    - ĐỒNG BỘ: Nếu ảnh nào không nhận diện được landmark, xóa ảnh đó khỏi data/processed/images
      để đảm bảo tính đồng bộ 100% giữa ảnh và đồ thị.
"""

import os
import cv2
import logging
import urllib.request
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Cấu hình log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Thiết lập đường dẫn
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
IMAGES_DIR = PROCESSED_DIR / "images"
GRAPHS_DIR = PROCESSED_DIR / "graphs"

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = PROJECT_ROOT / "face_landmarker.task"

STANDARD_LABELS = ['anger', 'happiness', 'neutral', 'sadness']

def download_model():
    """Tải file mô hình Face Landmarker nếu chưa có sẵn."""
    if not MODEL_PATH.exists():
        logger.info(f"Đang tải mô hình Face Landmarker từ: {MODEL_URL}")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            logger.info("Đã tải thành công mô hình Face Landmarker.")
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình: {e}")
            raise e
    else:
        logger.info("Đã tìm thấy mô hình Face Landmarker tại chỗ.")

def setup_directories():
    """Tạo sẵn các thư mục lưu đồ thị."""
    for label in STANDARD_LABELS:
        (GRAPHS_DIR / label).mkdir(parents=True, exist_ok=True)

def main():
    # Khởi tạo MediaPipe vision tasks
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError:
        logger.error("Không tìm thấy thư viện 'mediapipe'. Vui lòng cài đặt trước bằng lệnh: pip install mediapipe")
        return

    logger.info("BẮT ĐẦU TRÍCH XUẤT LANDMARKS BẰNG MEDIAPIPE TASKS API")
    download_model()
    setup_directories()

    # Cấu hình Face Landmarker
    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )

    stats = {'total': 0, 'success': 0, 'failed_deleted': 0}

    # Quét tất cả ảnh trong processed/images
    image_paths = []
    for label in STANDARD_LABELS:
        label_dir = IMAGES_DIR / label
        if label_dir.exists():
            image_paths.extend(list(label_dir.glob("*.*")))

    logger.info(f"Tổng số ảnh quét được trong processed/images: {len(image_paths)}")
    
    # Khởi tạo landmarker
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        for img_path in tqdm(image_paths, desc="Trích xuất landmarks", unit="ảnh"):
            stats['total'] += 1
            
            # Đọc ảnh grayscale
            img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                logger.warning(f"Không thể đọc ảnh: {img_path}. Tiến hành xóa.")
                img_path.unlink()
                stats['failed_deleted'] += 1
                continue
                
            # Chuyển sang RGB (yêu cầu của MediaPipe)
            img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
            
            # Chuyển đổi sang đối tượng mp.Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            
            # Chạy landmark detection
            detection_result = landmarker.detect(mp_image)
            
            if detection_result.face_landmarks:
                # Lấy các điểm mốc cho khuôn mặt đầu tiên
                face_landmarks = detection_result.face_landmarks[0]
                
                # Trích xuất tọa độ x, y chuẩn hóa
                landmarks = np.array([
                    [lm.x, lm.y] for lm in face_landmarks
                ], dtype=np.float32)
                
                # Cắt lấy chính xác 468 điểm (bỏ các điểm phụ thêm như con ngươi/iris nếu có)
                if len(landmarks) > 468:
                    landmarks = landmarks[:468]
                
                # Lưu landmark thành file .npy
                dest_npy_path = GRAPHS_DIR / img_path.parent.name / f"{img_path.stem}.npy"
                np.save(str(dest_npy_path), landmarks)
                stats['success'] += 1
            else:
                # Không nhận diện được mặt: Xóa ảnh nguồn để đảm bảo đồng bộ
                img_path.unlink()
                stats['failed_deleted'] += 1

    # Báo cáo tổng kết
    logger.info("="*50)
    logger.info("HOÀN TẤT TRÍCH XUẤT LANDMARKS!")
    logger.info(f"Tổng số ảnh xử lý  : {stats['total']}")
    logger.info(f"Trích xuất thành công: {stats['success']}")
    logger.info(f"Không nhận diện được & Đã xóa: {stats['failed_deleted']}")
    logger.info("="*50)

if __name__ == "__main__":
    main()
