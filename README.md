# 🛰️ Ships in Satellite Imagery Detection

![Sample Ship Detection](data/scenes/sfbay_1.png)  
*Example satellite image from San Francisco Bay*

A machine learning project to detect ships in satellite imagery using convolutional neural networks (CNNs). Includes dataset preprocessing, model training, and inference pipelines.

## 📦 Dataset
**Source:** [Kaggle - Ships in Satellite Imagery](https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery)  
**License:** CC-BY-SA-4.0  

### Included in this repo (`/data`):
- `shipsnet.json`  
  - 4,000 labeled 80×80 RGB images  
  - Labels: `1` (ship) / `0` (no ship)  
  - Metadata: Latitude, longitude, timestamp  
- `scenes/`  
  - 8 full-resolution satellite scenes (PNG):  
    - San Francisco Bay (4)  
    - Long Beach, CA (4)  

## 🛠️ Setup

### 1. Clone the repository

git clone https://github.com/yourusername/ships-in-satellite.git
cd ships-in-satellite

### 2. Install dependencies
pip install -r requirements.txt
(See requirements.txt for package versions)

## 🚀 Usage
Data Preprocessing
Convert shipsnet.json to images + CSV:

python scripts/preprocess_data.py

## Outputs:
data/images/ (individual ship patches)
data/labels.csv (metadata + coordinates)

## Train the model

python train.py \
  --epochs 50 \
  --batch_size 32 \
  --model resnet18
  
## Detect ships in new scenes
python detect.py --input data/scenes/sfbay_1.png --output results/

## 📊 Model Performance Comparison

| Model      | Accuracy | Precision | Recall |
|------------|----------|-----------|--------|
| ResNet18   | 96.2%    | 94.1%     | 97.8%  |
| Custom CNN | 92.4%    | 89.5%     | 93.2%  |

## 📂 Repository Structure

.
├── data/                   # Dataset (raw and processed)
├── notebooks/              # Jupyter notebooks for EDA
├── scripts/                # Utility scripts
│   ├── preprocess_data.py  # JSON → images converter
│   └── visualize.py        # Geo-plotting tools
├── models/                 # Saved model weights
├── train.py                # Training script
├── detect.py               # Inference script
└── requirements.txt        # Python dependencies

## 🤝 Contributing
Fork the project

Create a branch (git checkout -b feature/improve-model)

Submit a Pull Request

## 📜 License
CC-BY-SA-4.0 (inherited from Kaggle dataset).
For commercial use, check Kaggle's terms.
