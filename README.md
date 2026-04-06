# 🏥 Sepsis Prediction using LSTM

An AI-powered healthcare system for early detection of sepsis using deep learning and machine learning models on ICU time-series data.

🚨 Early detection can significantly improve survival rates.

# Sepsis Prediction Model using LSTM

A machine learning project for predicting sepsis in ICU patients using deep learning (LSTM) and traditional machine learning algorithms. This project aims to develop an accurate, early warning system for sepsis detection to improve patient outcomes.

## 📋 Project Overview

Sepsis is a life-threatening condition that occurs when the body's response to infection causes tissue damage. Early detection and intervention are critical for patient survival. This project implements various machine learning and deep learning models to predict sepsis onset based on clinical vital signs and laboratory values.

### Key Features
- **LSTM-based deep learning models** for sequential patient monitoring data
- **Multiple baseline models** including LightGBM and traditional ML algorithms
- **Data balancing techniques** including SMOTE for handling imbalanced datasets
- **Comprehensive data preprocessing** including interpolation and missing value imputation
- **Model evaluation metrics** including ROC-AUC, classification reports, and performance analysis

## 📁 Project Structure

```
sepsis-prediction/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
│
├── notebooks/                # Jupyter notebooks for development and training
│   ├── 01_SepsisModelTraining.ipynb      # Main model training notebook
│   ├── 02_SepsisModel_SMOTE.ipynb        # SMOTE-based model training
│   ├── 03_SepsisModel_Sample.ipynb       # Sample exploration and analysis
│   └── 04_Sepis.ipynb                    # Additional model experiments
│
├── models/                   # Trained model files
│   ├── lstm_model.keras                  # LSTM model (Keras format)
│   ├── lstm_model_no_optimizer.keras     # LSTM model without optimizer
│   ├── lstm_sepsis_model.h5              # LSTM model (H5 format)
│   └── lstm_model.h5                     # Alternative LSTM model
│
├── data/
│   ├── processed/            # Pre-processed and cleaned datasets
│   │   ├── balanced_training_data.csv       # Balanced dataset
│   │   └── balanced_training_data_smote.csv # SMOTE-balanced dataset
│   │
│   └── raw/                  # Original raw datasets
│       └── original_dataset.csv              # Original dataset from source
│
└── docs/                     # Documentation and problem statements
    ├── Problem_Statement.pdf      # Hackathon problem statement
    └── Sepsis_Overview.pdf        # Sepsis clinical overview
```

## 🚀 Getting Started

### Prerequisites
- Python 3.7 or higher
- pip or conda package manager
- Jupyter Notebook or JupyterLab

### Installation

1. **Clone or download the repository:**
   ```bash
   cd sepsis-prediction
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

## 📊 Dataset

### Data Sources
- **Original Dataset:** Located in `data/raw/original_dataset.csv`
- **Processed Datasets:** Located in `data/processed/`
  - `balanced_training_data.csv` - Standard balanced dataset
  - `balanced_training_data_smote.csv` - SMOTE-balanced dataset for handling class imbalance

### Features
The dataset contains vital signs and laboratory measurements from ICU patients including:
- Heart rate, blood pressure, temperature
- Oxygen saturation, respiratory rate
- Laboratory values, infection markers
- Patient demographics and outcome labels

### Data Preprocessing
- **Missing Value Imputation:** Uses forward fill, backward fill, and interpolation strategies
- **Feature Scaling:** StandardScaler normalization
- **Class Balancing:** SMOTE (Synthetic Minority Over-sampling Technique) for handling imbalanced data
- **Sequence Preparation:** Time-series data structured for LSTM models

## 🤖 Models

### Available Models

1. **LSTM Models** (Primary)
   - File: `lstm_model.keras` or `lstm_sepsis_model.h5`
   - Architecture: 2 LSTM layers with dropout and dense layers
   - Handles sequential temporal patient data
   - Best performance on time-series patterns

2. **LightGBM** (Baseline)
   - Gradient boosting model
   - Fast training and inference
   - Good baseline performance

3. **Additional Classifiers** (Experimentation)
   - Random Forest, Logistic Regression, SVM
   - Results documented in notebooks

### Model Performance
- Metrics tracked: ROC-AUC, Accuracy, Precision, Recall, F1-Score
- Cross-validation on training data (5-fold)
- Evaluation on held-out test sets

## 📈 Training & Results

### Training Notebooks

1. **01_SepsisModelTraining.ipynb** (Main)
   - Primary LSTM model training pipeline
   - Data loading and preprocessing
   - Model architecture and training
   - Performance evaluation

2. **02_SepsisModel_SMOTE.ipynb**
   - SMOTE-based data balancing
   - Improved handling of class imbalance
   - ROC-AUC and classification metrics

3. **03_SepsisModel_Sample.ipynb**
   - Exploratory data analysis
   - Sample predictions and analysis
   - Model interpretation

### Key Results
- Achieved high ROC-AUC scores on sepsis detection
- LSTM models capture temporal patterns effectively
- SMOTE balancing improves minority class recall
- Models ready for clinical validation

## 🛠 Usage

### Running a Trained Model

```python
import keras.models
import pandas as pd

# Load the model
model = keras.models.load_model('models/lstm_model.keras')

# Load and preprocess data
data = pd.read_csv('data/processed/balanced_training_data.csv')

# Make predictions
predictions = model.predict(data)
```

### Training a New Model

1. Open `notebooks/01_SepsisModelTraining.ipynb`
2. Update data path if needed
3. Run all cells to train model from scratch
4. Save the trained model for deployment

## 📋 Requirements

Core packages required (listed in `requirements.txt`):
- tensorflow/keras - Deep learning framework
- pandas - Data manipulation
- numpy - Numerical computing
- scikit-learn - Machine learning utilities
- lightgbm - Gradient boosting
- matplotlib/seaborn - Visualization
- jupyter - Interactive notebooks

## 🔍 Troubleshooting

### Common Issues

1. **Missing modules:** Run `pip install -r requirements.txt`
2. **CUDA/GPU issues:** CPU mode will be used automatically if GPU unavailable
3. **Path errors in notebooks:** Update data paths relative to notebook location

## 📝 References

- **Problem Statement:** See `docs/Problem_Statement.pdf`
- **Clinical Overview:** See `docs/Sepsis_Overview.pdf`
- **Original Dataset Source:** Documented in the notebooks
- **LSTM Architecture:** Based on standard sequence-to-sequence models

## 👥 Contributors

This project was developed as part of a Machine Learning Hackathon initiative focused on healthcare applications.

## 📄 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

To improve this project:
1. Review the current notebooks and results
2. Update models or try new architectures
3. Improve data preprocessing techniques
4. Enhance evaluation metrics and reporting

## 📧 Contact & Support

For questions about this project or technical issues, refer to the documentation files and code comments within the notebooks.

---

**Last Updated:** April 2026  
**Project Type:** Healthcare ML / Hackathon  
**Primary Model:** LSTM Neural Networks
