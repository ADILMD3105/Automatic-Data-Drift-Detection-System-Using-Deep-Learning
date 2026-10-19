# DriftDetect AI
### Automatic Data Drift Detection System Using Deep Learning

DriftDetect AI is an interactive, web-based data drift monitoring system that uses deep autoencoders to analyze datasets, evaluate reconstruction errors, and identify samples that deviate from a reference distribution.

Built with PyTorch and Flask, the application integrates a dynamic machine learning pipeline with a custom visual interface, bringing data ingestion, model training, drift analysis, and result visualization into a unified workflow.

## ✨ Key Features

- **Interactive Monitoring Interface** — A custom web interface for uploading datasets, initiating analysis, and exploring results.
- **Dynamic Autoencoder Training** — Trains a fresh autoencoder for each analysis using the uploaded reference dataset.
- **Automated Data Preprocessing** — Handles supported file formats, numeric conversion, missing values, feature filtering, and standardization.
- **Reconstruction-Error Analysis** — Evaluates reference and current samples using a trained autoencoder.
- **Statistical Thresholding** — Derives a detection threshold from the 95th percentile of reference reconstruction errors.
- **Sample-Level Drift Flags** — Identifies current samples whose reconstruction errors exceed the reference threshold.
- **Visual Results and Metrics** — Presents drift statistics, reconstruction errors, and analysis stages through the frontend.
- **Flask API Integration** — Connects the browser interface to the Python machine learning pipeline.

## 🧠 System Architecture

The application follows a modular architecture that separates the user interface, API layer, data processing, model training, and detection logic.

```text
Reference Dataset ──┐
                    ├──> Flask API
Current Dataset ────┘        │
                             ▼
                    Data Preprocessing
                             │
                             ▼
                  Reference-Based Scaling
                             │
                             ▼
                   Autoencoder Training
                    (Reference Only)
                             │
                             ▼
                  Reconstruction Errors
                    ┌────────┴────────┐
                    ▼                 ▼
              Reference Data     Current Data
                    │                 │
                    ▼                 ▼
                 P95 Threshold ──> Drift Flags
                                      │
                                      ▼
                             Interactive Results
```

## ⚙️ Technology Stack

| Component | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | PyTorch |
| Data Processing | Pandas, NumPy |
| Preprocessing | scikit-learn |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Data Formats | CSV, TXT, XLSX |

## 📁 Project Structure

```text
DriftDetect-AI/
├── index.html
├── server.py
├── requirements.txt
├── README.md
├── images/
└── src/
    ├── autoencoder.py
    ├── data_loader.py
    ├── detect_drift.py
    └── train.py
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or later
- pip
- A supported operating system with the required Python dependencies

### Installation

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd DriftDetect-AI
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Launch the Application

```bash
python server.py
```

Open the application in your browser:

**http://127.0.0.1:5000**

## 📊 How It Works

1. **Data Ingestion:** Upload reference and current datasets through the interface.
2. **Preprocessing:** Convert supported input data into a suitable numerical representation and standardize features using statistics fitted on the reference data.
3. **Model Training:** Train a fresh autoencoder using the reference dataset only.
4. **Reconstruction Analysis:** Calculate reconstruction errors for reference and current samples using the trained model.
5. **Threshold Calculation:** Determine the 95th-percentile reference reconstruction-error threshold.
6. **Drift Evaluation:** Flag current samples whose reconstruction errors exceed the threshold.
7. **Visualization:** Display the resulting metrics and analysis output through the web interface.

## 🔬 Detection Methodology

The system uses reconstruction error as an indicator of deviation from the reference distribution.

For a reference reconstruction-error distribution \(E_r\), the threshold is:

\[
T = P_{95}(E_r)
\]

For each current sample with reconstruction error \(e_i\):

\[
\operatorname{flag}_i =
\begin{cases}
1 & e_i > T\\
0 & e_i \leq T
\end{cases}
\]

The drift ratio is calculated as the proportion of current samples flagged by this rule.

The approach is unsupervised and does not require labeled drift examples. Its effectiveness depends on the characteristics of the input data, preprocessing, and model behavior.

## 🛠️ Design Philosophy

DriftDetect AI combines a modular Python backend with a purpose-built visual interface. Its design emphasizes a clear workflow from dataset ingestion through computation to interpretable monitoring results.

The separation between the frontend, Flask API, data-processing utilities, and autoencoder pipeline supports maintainability and further experimentation.

## 🔮 Future Development

Potential extensions include streaming-data support, more extensive drift benchmarks, training-progress visualization, and additional statistical detection methods.

## 📄 License

Add an appropriate open-source license before distributing the project for reuse.
