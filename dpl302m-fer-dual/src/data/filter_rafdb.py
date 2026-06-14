"""
src/data/filter_rafdb.py
------------------------
Lọc bộ dữ liệu RAF-DB: chỉ giữ lại 4 nhãn cảm xúc cần thiết.

Mapping nhãn RAF-DB gốc:
  1 - Surprise  ❌ Xóa
  2 - Fear      ❌ Xóa
  3 - Disgust   ❌ Xóa
  4 - Happiness ✅ Giữ lại
  5 - Sadness   ✅ Giữ lại
  6 - Anger     ✅ Giữ lại
  7 - Neutral   ✅ Giữ lại

Quy trình:
  1. Xóa toàn bộ folder 1/, 2/, 3/ trong DATASET/train/ và DATASET/test/
  2. Lọc file train_labels.csv và test_labels.csv, chỉ giữ dòng có label 4-7
  3. In thống kê chi tiết trước và sau khi xử lý

Cách chạy:
  python -m src.data.filter_rafdb
  hoặc: python src/data/filter_rafdb.py
"""

import os
import shutil
import pandas as pd

# ============================================================
# CẤU HÌNH
# ============================================================

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to original raw dataset (unchanged)
RAW_RAFDB = os.path.normpath(os.path.join(ROOT_DIR, "..", "..", "data", "raw", "raf-db"))
# Path where we store the processed version (preprocessed)
PREPROCESSED_RAFDB = os.path.normpath(os.path.join(ROOT_DIR, "..", "..", "data", "preprocessed", "raf-db"))

DATASET_DIR     = os.path.join(PREPROCESSED_RAFDB, "DATASET")
TRAIN_IMG_DIR   = os.path.join(DATASET_DIR, "train")
TEST_IMG_DIR    = os.path.join(DATASET_DIR, "test")
TRAIN_LABEL_CSV = os.path.join(PREPROCESSED_RAFDB, "train_labels.csv")
TEST_LABEL_CSV  = os.path.join(PREPROCESSED_RAFDB, "test_labels.csv")

# Nhãn muốn GIỮ LẠI
KEEP_LABELS = {4, 5, 6, 7}

# Nhãn muốn XÓA
REMOVE_LABELS = {1, 2, 3}

# Tên cảm xúc để hiển thị log
LABEL_NAMES = {
    1: "Surprise",
    2: "Fear",
    3: "Disgust",
    4: "Happiness",
    5: "Sadness",
    6: "Anger",
    7: "Neutral",
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_header(text: str):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def count_images_in_folder(base_dir: str) -> dict:
    """Đếm số ảnh trong từng subfolder nhãn (1-7)."""
    counts = {}
    for label_id in sorted(os.listdir(base_dir)):
        folder = os.path.join(base_dir, label_id)
        if os.path.isdir(folder) and label_id.isdigit():
            n = len([f for f in os.listdir(folder) if not f.startswith(".")])
            counts[int(label_id)] = n
    return counts


def print_distribution(counts: dict, title: str):
    """In bảng phân phối nhãn."""
    print(f"\n{title}")
    print(f"  {'Nhãn':<5} {'Cảm xúc':<12} {'Số ảnh':>8}  {'Trạng thái'}")
    print(f"  {'-'*45}")
    total = 0
    for label_id, count in sorted(counts.items()):
        name = LABEL_NAMES.get(label_id, "???")
        status = "Giữ lại" if label_id in KEEP_LABELS else "Xóa"
        print(f"  {label_id:<5} {name:<12} {count:>8}  {status}")
        total += count
    print(f"  {'-'*45}")
    print(f"  {'TOTAL':<18} {total:>8}")


def delete_label_folders(base_dir: str, labels_to_remove: set, split: str):
    """Xóa các folder nhãn không cần thiết."""
    deleted_total = 0
    for label_id in sorted(labels_to_remove):
        folder = os.path.join(base_dir, str(label_id))
        if os.path.exists(folder):
            img_count = len([f for f in os.listdir(folder) if not f.startswith(".")])
            shutil.rmtree(folder)
            print(f"  🗑️  [{split}] Đã xóa folder {label_id}/ ({LABEL_NAMES[label_id]}) — {img_count} ảnh")
            deleted_total += img_count
        else:
            print(f"  ⚠️  [{split}] Folder {label_id}/ không tồn tại, bỏ qua.")
    return deleted_total


def filter_label_csv(csv_path: str, keep_labels: set, split: str) -> pd.DataFrame:
    """Lọc file CSV chỉ giữ các dòng có nhãn trong keep_labels."""
    df = pd.read_csv(csv_path)
    before = len(df)

    df_filtered = df[df["label"].isin(keep_labels)].reset_index(drop=True)
    after = len(df_filtered)
    removed = before - after

    print(f"\n  📄 [{split}] {os.path.basename(csv_path)}")
    print(f"     Trước lọc : {before:>6} dòng")
    print(f"     Sau lọc   : {after:>6} dòng")
    print(f"     Đã xóa    : {removed:>6} dòng")

    # Lưu đè file gốc
    df_filtered.to_csv(csv_path, index=False)
    print(f"     ✅ Đã lưu lại: {csv_path}")

    return df_filtered


# ============================================================
# MAIN PIPELINE
# ============================================================

def run():
    print_header("RAF-DB DATASET FILTER — Giữ lại 4 nhãn: Happiness, Sadness, Anger, Neutral")

    # 0. Copy raw dataset to preprocessed (overwrite any previous processed version)
    if os.path.exists(PREPROCESSED_RAFDB):
        shutil.rmtree(PREPROCESSED_RAFDB)
    shutil.copytree(RAW_RAFDB, PREPROCESSED_RAFDB)
    print(f"\n  📁 Đã sao chép RAW -> PREPROCESSED: {PREPROCESSED_RAFDB}")

    # 1. Kiểm tra đường dẫn của dữ liệu đã sao chép
    for path in [TRAIN_IMG_DIR, TEST_IMG_DIR, TRAIN_LABEL_CSV, TEST_LABEL_CSV]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy: {path}")
    print(f"\n  ✅ Đã xác nhận đường dẫn RAF-DB (preprocessed): {PREPROCESSED_RAFDB}")

    # 2. Thống kê TRƯỚC khi xóa
    print_header("THỐNG KÊ TRƯỚC KHI LỌC")
    train_counts_before = count_images_in_folder(TRAIN_IMG_DIR)
    test_counts_before  = count_images_in_folder(TEST_IMG_DIR)
    print_distribution(train_counts_before, "TRAIN")
    print_distribution(test_counts_before,  "TEST")

    # 3. Xác nhận trước khi xóa
    print("\n" + "=" * 60)
    labels_to_remove_names = [f"{k} ({LABEL_NAMES[k]})" for k in sorted(REMOVE_LABELS)]
    print(f"  ⚠️  Sắp xóa vĩnh viễn nhãn: {', '.join(labels_to_remove_names)}")
    answer = input("  Bạn có chắc chắn muốn tiếp tục? (yes/no): ").strip().lower()

    if answer != "yes":
        print("  ❌ Đã hủy thao tác. Không có gì bị thay đổi.")
        return

    # 4. Xóa folder ảnh
    print_header("XÓA FOLDER ẢNH")
    del_train = delete_label_folders(TRAIN_IMG_DIR, REMOVE_LABELS, "train")
    del_test  = delete_label_folders(TEST_IMG_DIR,  REMOVE_LABELS, "test")

    # 5. Lọc file label CSV
    print_header("LỌC FILE LABEL CSV")
    filter_label_csv(TRAIN_LABEL_CSV, KEEP_LABELS, "train")
    filter_label_csv(TEST_LABEL_CSV,  KEEP_LABELS, "test")

    # 6. Thống kê SAU khi xóa
    print_header("THỐNG KÊ SAU KHI LỌC")
    train_counts_after = count_images_in_folder(TRAIN_IMG_DIR)
    test_counts_after  = count_images_in_folder(TEST_IMG_DIR)
    print_distribution(train_counts_after, "TRAIN")
    print_distribution(test_counts_after,  "TEST")

    # 7. Tóm tắt
    print_header("TÓM TẮT")
    total_train = sum(train_counts_after.values())
    total_test  = sum(test_counts_after.values())
    print(f"  🖼️  Tổng ảnh TRAIN còn lại : {total_train}")
    print(f"  🖼️  Tổng ảnh TEST còn lại  : {total_test}")
    print(f"  🗑️  Tổng ảnh đã xóa        : {del_train + del_test}")
    print(f"\n  ✅ Hoàn tất! RAF-DB đã được lọc còn 4 nhãn.")
    print("=" * 60)


if __name__ == "__main__":
    run()
