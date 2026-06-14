# Facial Emotion Recognition

A Deep Learning project implementing a **Dual-Branch GCN-CNN** (Graph Convolutional Network and Convolutional Neural Network) architecture to classify 4 basic human emotion categories: **Anger, Happiness, Neutral, and Sadness**. The architecture fuses geometric facial landmark coordinates (GCN branch) with facial texture features (CNN branch) for highly robust facial expression recognition.

For detailed design specifications, see the [Refactoring Master Guideline](doc/refine-data-plan.md).

---

## 1. Prerequisites
    
-   **Language:** Python 3.8 -> 3.11 (Python 3.10 or 3.11 recommended).
        

## 2. Environment Setup

To avoid library conflicts with other projects on your machine, it is recommended to create a virtual environment.

### Step 2.1: Clone/Download the source code

Open Terminal (or Command Prompt / PowerShell) and navigate to the project directory:

```
cd project-path
```

### Step 2.2: Create a virtual environment (venv)

Use Python's default `venv` module to create a virtual environment named `venv`:

```
python -m venv venv
```

_(After running this command, you will see a new folder named `venv` appear in the project root directory)._

### Step 2.3: Activate the virtual environment

Depending on your operating system, run the corresponding command:

-   **For Windows (Command Prompt):**
    
    ```cmd
    venv\Scripts\activate.bat
    ```
    
-   **For Windows (PowerShell):**
    
    ```powershell
    .\venv\Scripts\Activate.ps1
    ```
    
-   **For macOS and Linux:**
    
    ```bash
    source venv/bin/activate
    ```
    

_(Success indicator: You should see `(venv)` prepended to your Terminal prompt)._

### Step 2.4: Install required libraries

With the virtual environment activated, proceed to install all dependencies from the `requirements.txt` file:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Run the Data Preprocessing Pipeline

Before running notebooks or training models, you need to preprocess the raw data into the final processed format. Run the following scripts in order within your Terminal:

```bash
# 1. Standardize and rename the RAF-DB dataset
python src/data/rename_rafdb.py

# 2. Convert images to Grayscale and resize them into 2 branches (48x48 and 224x224)
python src/data/format_data.py

# 3. Split dataset (Train/Val/Test) and balance the data (Over/Undersampling)
python src/data/split_balance.py
```

_Result:_ The `data/final/` directory will be created with a standardized structure, ready for training.

> [!NOTE]
> For a deep dive into the preprocessing logic, crop strategies, landmark extraction via MediaPipe, and train set balancing (oversampling/undersampling to 4,500 samples per class), refer to the [Data Processing Plan](doc/data-processing-plan.md) and the [System Pipeline Plan](doc/pipeline-plan.md).

---

## 4. Notebook Execution Guide (EDA & Sanity Check)

The project provides 2 predefined Notebook files written in the Python Script format (`# %%`), which is easier to manage with Git than traditional `.ipynb` files.

### Method 1: Run directly in Visual Studio Code (Highly Recommended)

1.  Install the following extensions in VS Code: **Jupyter** and **Python**.
    
2.  Open `notebooks/1.0-source-datasets-exploration.py` or `2.0-final-data-sanity-check.py`.
    
3.  Click the `Run Cell` button (displayed subtly above each `# %%` comment line) or click `Run Below` to execute the entire file.
    
4.  Select the virtual environment `venv` as the kernel when prompted by VS Code. The visualizations/plots will be displayed interactively in the Interactive Window.
    

### Method 2: Run using traditional Jupyter Notebook

If you prefer using the Jupyter web interface:

1.  Ensure you have activated `(venv)`.
    
2.  Run the command to start Jupyter Notebook:
    
    ```bash
    jupyter notebook
    ```
    
3.  The web browser will open automatically. Navigate to the `notebooks/` directory and open the `.py` files or convert them to `.ipynb` to execute them.
    

> [!NOTE]
> To understand the analytical goals of the Exploratory Data Analysis and sanity checks (such as verifying dataset proportions, ensuring 1-to-1 sync between images and landmarks, and checking dataloader dimensions), see the [EDA Plan](doc/eda-plan.md).

---

## 5. Documentation & Deep Dive

For a detailed understanding of the system's design, pipeline architecture, and training process, refer to the following documentation files located in the `doc/` directory:

-   **[pipeline-plan.md](doc/pipeline-plan.md) (System Architecture & Pipeline)**: Details the overall end-to-end design divided into 4 key stages: Ingestion/Filtration, Model Training, Evaluation & Analysis, and Real-time Deployment POC.
-   **[data-processing-plan.md](doc/data-processing-plan.md) (Data Formatting & Processing)**: Explains the preprocessing workflow for RAF-DB, FER2013, and CK+ datasets. Discusses the Grayscale 224x224 texture branch (CNN) and the 468-facial landmark geometric branch (GCN).
-   **[eda-plan.md](doc/eda-plan.md) (EDA & Sanity Verification)**: Details the goals for dataset distribution analysis, qualitative visual inspections, pixel intensity checks, and data pipeline verification tests.
-   **[training-plan.md](doc/training-plan.md) (Modeling Strategy & Evaluation)**: Details the design of the baseline MobileNetV2 (Image-only) model, the proposed Dual-Branch GCN-CNN architecture, common hyperparameters, and the evaluation metrics used (F1-score, Confusion Matrix, FPS).
-   **[colab-training-guide.md](doc/colab-training-guide.md) (Colab GPU Training Guide)**: A tutorial on formatting code/data for Google Drive, mounting Drive inside Colab, setting up high-speed local SSD directories, and configuring Keras callbacks.
-   **[refine-data-plan.md](doc/refine-data-plan.md) (Refactoring Master Guideline)**: Serves as a master reference for developers, documenting implementation constraints, directory architectures, class balance synchronization rules, and 1-channel vs. 3-channel input workarounds.

---

## 6. Project Structure

```
├── data/
│   ├── raw/                # Original dataset downloaded from Kaggle
│   ├── preprocessed/       # Preprocessed data with grouped labels
│   ├── processed/          # Data converted to Grayscale & resized
│   └── final/              # Data split and balanced
├── notebooks/              
│   ├── 1.0-source-datasets-exploration.py  # Source EDA (Histogram, Crosstab, Outliers)
│   └── 2.0-final-data-sanity-check.py      # Pipeline Validation, Data Leakage check
├── src/
│   ├── data/
│   │   ├── rename_rafdb.py
│   │   ├── format_data.py
│   │   ├── split_balance.py
│   │   └── data_loader.py  # Module to feed data to GPU
│   └── models/             # (Upcoming) Code for training Baseline and Transfer Learning models
├── doc/
│   ├── colab-training-guide.md   # Google Colab environment setup & training callbacks configuration
│   ├── data-processing-plan.md   # Detailed data formatting, landmark extraction, and dataset balancing
│   ├── eda-plan.md               # Source dataset analysis plan & final data loader sanity checks
│   ├── pipeline-plan.md          # End-to-end system stages from ingestion to real-time inference POC
│   ├── refine-data-plan.md       # Refactoring master guideline for dual-branch dataset synchronization
│   └── training-plan.md          # Baseline MobileNetV2 & Dual-Branch GCN-CNN modeling strategy
├── requirements.txt        # List of libraries to install
└── README.md               # This instructions file
```

_If you encounter any errors during installation, please verify that your Terminal prompt is prefixed with `(venv)`!_
