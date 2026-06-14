
# MASTER GUIDELINE: REFACTORING DATA PIPELINE & REPORT 2

**Project:** Empirical Research and Integration of Dual-Branch GCN-CNN Architecture for FER.

**Target Audience:** Developer & AI Coding Agent in VSCode.

## 🟢 STATUS: FULLY IMPLEMENTED (ĐÃ HOÀN THÀNH)

Kế hoạch tinh chỉnh dữ liệu này đã được thực thi và xác minh thành công trong mã nguồn dự án:
-   **Định dạng hình ảnh (224x224 Grayscale):** Đã triển khai tại [format_data.py](file:///Users/anhnn/Documents/lectures/mentor-fptu/support/sem-09/dpl302m-fer-dual/src/data/format_data.py).
-   **Trích xuất Landmark Đồ thị (468 points, `.npy`):** Đã triển khai tại [extract_landmarks.py](file:///Users/anhnn/Documents/lectures/mentor-fptu/support/sem-09/dpl302m-fer-dual/src/features/extract_landmarks.py) sử dụng MediaPipe Tasks API.
-   **Phân chia & Cân bằng đồng bộ (Stratified 70-15-15, 4500 mẫu/nhãn train):** Đã triển khai tại [split_balance.py](file:///Users/anhnn/Documents/lectures/mentor-fptu/support/sem-09/dpl302m-fer-dual/src/data/split_balance.py).
-   **Custom DataLoader (nạp song song Image & Graph):** Đã triển khai tại [data_loader.py](file:///Users/anhnn/Documents/lectures/mentor-fptu/support/sem-09/dpl302m-fer-dual/src/data/data_loader.py).
-   **Kiểm tra và kiểm định chất lượng (Sanity Check):** Đã kiểm định thông qua notebook [2.0-final-data-sanity-check.ipynb](file:///Users/anhnn/Documents/lectures/mentor-fptu/support/sem-09/dpl302m-fer-dual/notebooks/2.0-final-data-sanity-check.ipynb).
-   **Huấn luyện Baseline (MobileNetV2 image branch):** Đã triển khai tại [train_baseline.py](file:///Users/anhnn/Documents/lectures/mentor-fptu/support/sem-09/dpl302m-fer-dual/src/models/train_baseline.py).

## CORE DIRECTIVES (MANDATORY)

1.  **Language:** The final report02 **MUST BE WRITTEN IN ENGLISH 100%**. (Requirement from Instructor's rubric).
    
2.  **Architecture Pivot:** Abandon the pure CNN/Transfer Learning approach. The project is now a **Dual-Branch GCN-CNN** (Geometric + Texture).
    
3.  **Image Format:** All images MUST be converted to **224x224, 1-Channel (Grayscale)**. Abandon the 48x48 format completely.
    
4.  **Omit Related Works:** Do not include a "Related Works" section in Report 2 (already handled in Report 1).
    

## SECTION-BY-SECTION REFACTORING PLAN (From Old Draft to New Report 2)

### 1. Introduction & EDA (Exploratory Data Analysis)

-   **What to KEEP (Reuse):**
    
    -   The core logic of merging RAF-DB, FER2013, and CK+.
        
    -   Filtering down to exactly 4 classes: Anger, Happiness, Neutral, Sadness.
        
    -   The insightful EDA observations: Dataset imbalance (FER dominant) and Class imbalance (Happiness heavily dominant).
        
-   **What to CHANGE:**
    
    -   Translate all Vietnamese text to academic English.
        
    -   Ensure plots/charts are referenced (e.g., "As shown in Figure 1...").
        

### 2. Data Formatting (Preprocessed -> Processed)

-   **What to DISCARD:** Delete the logic/code that splits images into `baseline/` (48x48) and `transfer_learning/` (224x224).
    
-   **What to IMPLEMENT (New Logic):**
    
    -   Create **Branch 1 (Images):** Resize ALL images to `224x224` and convert to `Grayscale (1-channel)`. Reason: Facial expressions rely on contrast (wrinkles, edges), not color. Grayscale reduces compute/memory by 3x.
        
    -   Create **Branch 2 (Graphs):** Introduce a script using `MediaPipe Face Mesh` to extract $(x,y)$ landmark coordinates from the processed images. Save them as `.npy` matrices.
        

### 3. Splitting & Balancing (Processed -> Final)

-   **What to KEEP:**
    
    -   The `train_test_split` logic (70% Train, 15% Val, 15% Test) using `stratify`.
        
    -   The logic of balancing **ONLY the Training set** (Undersampling Happiness/Neutral, Oversampling Anger/Sadness).
        
-   **What to CHANGE (Crucial Synchronization Rule):**
    
    -   Instruct the AI Agent: The splitting and balancing scripts MUST synchronize the `images/` folder and `graphs/` folder. If `img_001.jpg` is oversampled, `graph_001.npy` must also be duplicated.
        

### 4. DataLoader & Augmentation

-   **What to DISCARD:** Remove Keras `ImageDataGenerator`. It cannot handle loading an image and a `.npy` file simultaneously for a dual-branch network.
    
-   **What to IMPLEMENT:** Write a Custom Dataset class (e.g., PyTorch `Dataset` or `tf.data.Dataset`).
    
    -   _Logic:_ For index `i`, load both the Grayscale image tensor and the `.npy` graph tensor, returning a tuple: `((Image_Tensor, Graph_Tensor), Label)`.
        

### 5. ADD NEW SECTION: Initial Results (Baseline Training)

-   **Requirement:** The rubric demands "Initial results (minimum 1 model)".
    
-   **Action Plan:** Train a lightweight Transfer Learning model (e.g., MobileNetV2) using ONLY the Grayscale Image branch for 10-15 epochs.
    
-   **Include in Report:** Training Loss/Accuracy curves and a Confusion Matrix on the Test set.
    

## 🟢 TECHNICAL IMPLEMENTATION DETAILS (FOR AI AGENT)

Dear AI Agent, when modifying the source code (`src/`), adhere to these technical constraints:

### Directory Structure to Enforce:

```
data/
├── raw/                  # Original downloaded datasets
├── preprocessed/         # Filtered to 4 classes, unified naming
├── processed/
│   ├── images/           # All 224x224 Grayscale images
│   └── graphs/           # All .npy MediaPipe landmarks
└── final/
    ├── train/            # Balanced (images & graphs)
    ├── val/              # Natural distribution (images & graphs)
    └── test/             # Natural distribution (images & graphs)

```

### The "1-Channel Transfer Learning" Problem

We are using Grayscale (1-channel) images, but standard pre-trained models (MobileNet, ResNet) expect RGB (3-channels). **Implement one of the following solutions in the code:**

**Solution A (Preferred - PyTorch Framework):**

Modify the first convolutional layer of the pre-trained model to accept 1 channel instead of 3, keeping original weights by summing or averaging them across the channel dimension, or initializing a new Conv2d layer.

```
# PyTorch Example
import torchvision.models as models
import torch.nn as nn

model = models.mobilenet_v2(pretrained=True)
original_conv = model.features[0][0] 
model.features[0][0] = nn.Conv2d(
    in_channels=1, # Changed from 3 to 1
    out_channels=original_conv.out_channels,
    kernel_size=original_conv.kernel_size,
    stride=original_conv.stride,
    padding=original_conv.padding,
    bias=False
)

```

**Solution B (TensorFlow/Keras Framework):**

Duplicate the 1 grayscale channel into 3 identical channels within the DataLoader/Data Pipeline before passing to the model.

```
# TensorFlow Example inside mapping function
import tensorflow as tf

def load_and_preprocess(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=1) # Read as Grayscale
    img = tf.image.resize(img, [224, 224])
    img = img / 255.0 # Normalize
    img = tf.image.grayscale_to_rgb(img) # Convert [224,224,1] to [224,224,3]
    return img, label

```

**End of Guideline.** _User Action:_ Please run the baseline training script based on these specs to generate the Loss/Accuracy charts, then provide them to finalize the text for Report 2.
