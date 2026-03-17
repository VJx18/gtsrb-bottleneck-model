# GTSRB Concept Bottleneck Model (CBM)
This project implements a **Concept Bottleneck Model (CBM)** for the German Traffic Sign Recognition Benchmark (GTSRB). Unlike traditional "black-box" Deep Learning models, this architecture improves interpretability by splitting the classification process into two sequential stages:

1.  **Concept Prediction:** The model first predicts human-interpretable concepts (e.g., "is triangular", "has red border", "contains arrow") from the input image.
2.  **Label Classification:** The final traffic sign class is predicted solely based on these intermediate concepts.

This project emphasizes software engineering best practices, including modular code organization, configuration management via dataclasses, version control, and reproducibility.

## 📂 Project Structure
#### The repository is organized as follows:
```plaintext
.
├── data/                       # Dataset directory (git-ignored)
│   ├── GTSRB/                  # Raw image data
│   │   ├── Final_Training/     # Training images folder
│   │   └── Final_Test/         # Test images folder
│   ├── concept_per_class.csv   # Concept annotations mapping classes to attributes
│   └── processed/              # (Optional) Cache for preprocessed tensors
├── experiments/                # Training artifacts
│   └── checkpoints/            # Directory where trained models (.pth) will be saved
├── src/                        # Source code package
│   ├── config/                 # Configuration management (dataclasses)
│   ├── data/                   # Custom Dataset and DataLoader implementations
│   ├── evaluation/             # Metric calculations and visualization logic
│   ├── models/                 # Neural network architectures (Stage 1 & Stage 2)
│   ├── training/               # Training loops and sequential training logic
│   └── utils/                  # Helper utilities (EarlyStopping, Metrics)
├── tests/                      # Unit tests for project modules
├── README.md                   # Project documentation
├── evaluate.py                 # CLI entry point for evaluation
├── requirements.txt            # Python dependencies
└── train.py                    # CLI entry point for training 
```

## 🚀 Installation & Setup
### 0. Prerequisites:
- **Python 3.10+**
- **CUDA-capable GPU** (recommended for training, but runs on CPU)

### 1. Clone the Repository
```bash
git clone [https://github.com/VJx18/gtsrb-bottleneck-model.git](https://github.com/VJx18/gtsrb-bottleneck-model.git)
cd gtsrb-bottleneck-model
```

### 2. Install Dependencies
It is recommended to use a virtual environment (venv or conda).
```bash
# Create venv (optional)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## 💾 Data Acquisition & Setup
To run the model, you need the GTSRB dataset and the provided concept annotations.

### 1. Download GTSRB Data:
- Download the **GTSRB Final Training Images** and **GTSRB Final Test Images** from the official [GTSRB Website](https://benchmark.ini.rub.de/gtsrb_dataset.html).
- Extract the contents into the `data/GTSRB` directory.

### 2. Concept Annotations:
- Place the `concept_per_class.csv` file directly in the `data/` folder. This file maps each of the 43 classes to binary feature vectors.

**Expected File Hierarchy:**
```plaintext
data/
├── concept_per_class.csv
└── GTSRB/
    ├── Final_Training/
    │   └── Images/
    │       ├── 00000/
    │       │   ├── 00000_00000.ppm
    │       │   └── ...
    │       └── ...
    └── Final_Test/
        └── Images/
            ├── 00000.ppm
            └── ...
```

## 🛠 Usage
The project provides Command Line Interfaces (CLI) for both training and evaluation.

### Training (End-to-End)
To train the complete CBM pipeline sequentially, run the following command:
```bash
python train.py --epochs 60 --lr 0.001 --seed 42
```

### What happens during training?
1. **Stage 1:** The `ConceptPredictor` is trained to map images to binary concepts using `BCEWithLogitsLoss`.
2. **Freezing:** The weights of Stage 1 are frozen to ensure the "bottleneck" structure.
3. **Stage 2:** The `LabelPredictor` is trained to map the predicted concepts to the final 43 traffic sign labels using `CrossEntropyLoss`.
4. **Artifacts**: The best model weights are saved to `experiments/checkpoints/`.

**CLI Arguments:**
- `--config`: Path to the config file (default: `src/config/config.py`).
- `--epochs`: Number of epochs (overrides config).
- `--lr`: Learning rate (overrides config).
- `--seed`: Random seed for reproducibility.

### Evaluation
To evaluate a trained model on the test set and generate concept-level metrics:
```bash
python evaluate.py --checkpoint experiments/checkpoints/best_cbm_model.pth
```

**CLI Arguments:**
- `--checkpoint`: Path to the .pth model file.
- `--data_path_testing`: (Optional) Custom path to test images if different from config.

**Output:** The script prints accuracy per concept, average precision/recall/F1-score for concepts, and final classification accuracy.

## 🧠 Model Architecture
The Concept Bottleneck Model consists of two distinct modules:

### 1. Concept Predictor (Stage 1)
- **Backbone:** `EfficientNetV2-S` (Pretrained on ImageNet).
- **Modifications:** The classification head is replaced with a fully connected layer outputting `num_concepts` logits.
- **Activation:** Sigmoid (to represent probability of concept presence).

### 2. Label Predictor (Stage 2)
- **Input:** Binary concept probabilities from Stage 1.
- **Architecture:** Multi-Layer Perceptron (MLP).
  - Structure: `Linear(num_concepts, 256) -> ReLU -> Dropout -> Linear(256, 128) -> ReLU -> Linear(128, 43)`.

## ⚙️ Configuration
All hyperparameters are managed via Python Dataclasses in `src/config/config.py`. You can modify default settings directly in the file, including:
- **DatasetConfig:** Paths, image size, batch size.
- **ModelConfig:** Backbone selection, dropout rates.
- **TrainingConfig:** Learning rate, patience (early stopping), device selection.

## 👥 Authors
Vraj Vijaybhai Patel, Tom Blum, Jakob Krappe  
Semesterprojekt-Gruppe 4  
Computer Science Department  
Humboldt-Universität zu Berlin