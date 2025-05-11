# THREAT_LOCK_LINK_RISK_ASSESSMENT

This project is designed to assess the risk level of web links using machine learning models. It helps detect potentially malicious or unsafe URLs, enabling users or systems to avoid threats such as phishing, malware, or spam.

## 🔍 Features

- URL-based risk classification
- Support for large-scale datasets
- Integration-ready model for real-time usage (e.g., browser extensions, backend services)

## 📁 Project Structure

```
THREAT_LOCK_LINK_RISK_ASSESSMENT/
│
├── data/                   # Dataset folder (not included in repo)
├── models/                 # Trained model files and scripts
├── notebooks/              # Jupyter notebooks for EDA and training
├── scripts/                # Python scripts for preprocessing, training, etc.
├── requirements.txt        # Dependencies
└── README.md               # Project overview
```

> 📌 **Note:** The dataset is not included in this repository due to GitHub's 100MB file size limit. You can download it from the link below.

## 📥 Download Dataset

📎 [Click here to download the dataset (Google Drive)](https://drive.google.com/file/d/1G07BN_oS8suIEQxMW2o4OaPKv3BpywqK/view?usp=drive_link)

After downloading, place the dataset file (e.g., `dataset1.csv`) in the `data/` directory.

## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/Yuvanraj-K-S/THREAT_LOCK_LINK_RISK_ASSESSMENT.git
cd THREAT_LOCK_LINK_RISK_ASSESSMENT
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

You can run the model or test predictions using the provided scripts or Jupyter notebooks in the `notebooks/` folder.

Example:

```bash
python scripts/predict.py --url "http://example.com"
```

## 📊 Model Training

To train the model from scratch using the dataset:

```bash
python scripts/train_model.py --data data/dataset1.csv
```

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
