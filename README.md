# ⚙️ AWS MLOps Pipeline

> End-to-end ML pipeline orchestrated with SageMaker & Step Functions — from preprocessing to deployment with CI/CD.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![AWS](https://img.shields.io/badge/AWS-SageMaker-FF9900?logo=amazonaws)](https://aws.amazon.com/sagemaker/)

## The Problem

Deploying ML models manually is error-prone and doesn't scale. Data scientists build models in notebooks, but getting them to production requires reproducible pipelines with automated preprocessing, training, evaluation, and deployment.

## The Solution

A fully automated MLOps pipeline that orchestrates the entire ML lifecycle on AWS. Push code → GitHub Actions triggers → Step Functions orchestrates → SageMaker trains & deploys. One pipeline, zero manual steps.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GitHub Actions CI/CD                       │
│                  (triggered on push/tag)                      │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              AWS Step Functions Workflow                      │
│                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
│  │  Preprocess   │──▶│    Train     │──▶│  Save Model  │    │
│  │ (SKLearn      │   │ (HuggingFace │   │              │    │
│  │  Processor)   │   │  Estimator)  │   │              │    │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘    │
│         │ ✗ Fail           │ ✗ Fail            │ ✗ Fail     │
│         ▼                  ▼                   ▼            │
│     [Error State]     [Error State]       [Error State]     │
│                                                              │
│  ┌──────────────┐   ┌──────────────┐                        │
│  │  Endpoint     │──▶│   Deploy     │                        │
│  │  Config       │   │  Endpoint    │                        │
│  └──────────────┘   └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
              ┌──────────────┐
              │  SageMaker    │
              │  Endpoint     │
              │  (Inference)  │
              └──────────────┘
```

## Key Features

- **Step Functions Orchestration** — Each pipeline step (preprocess, train, deploy) is a managed state with automatic error handling and catch states
- **HuggingFace Estimator** — Multi-label text classification with DistilBERT, early stopping, and checkpoint resumption
- **Automated Evaluation** — F1, accuracy, ROC-AUC, precision, and recall computed at each training run
- **CI/CD Ready** — GitHub Actions workflow triggers the pipeline on code push
- **Config-Driven** — OmegaConf + YAML configuration for S3 paths, hyperparameters, and infrastructure settings

## Pipeline Steps

| Step | Component | What It Does |
|------|-----------|-------------|
| **Preprocess** | `SKLearnProcessor` | Loads CSV, selects features, splits train/test, uploads to S3 |
| **Train** | `HuggingFace Estimator` | Fine-tunes DistilBERT for multi-label classification with early stopping |
| **Save Model** | `ModelStep` | Registers the trained model artifact in SageMaker |
| **Configure** | `EndpointConfigStep` | Creates endpoint configuration (instance type, count) |
| **Deploy** | `EndpointStep` | Deploys model as a real-time SageMaker endpoint |

## Quick Start

### Prerequisites

- Python 3.8+
- AWS account with SageMaker and Step Functions permissions
- Configured AWS credentials (`aws configure`)

### Installation

```bash
git clone https://github.com/aazizisoufiane/Aws_MlOps.git
cd Aws_MlOps
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your AWS role ARN and credentials
```

Edit `config/orchestrator.yaml` with your S3 bucket and paths:

```yaml
s3:
  bucket: your-s3-bucket-name
  input: aws_mlOps/input/data/train.csv
  prefix: aws_mlOps/sagemaker-pipeline/stepfunctions
region: us-east-1
```

### Run

Open `orchestrator.ipynb` and execute cells sequentially, or convert to a Python script for CI/CD:

```bash
jupyter nbconvert --to script orchestrator.ipynb
python orchestrator.py
```

## Project Structure

```
Aws_MlOps/
├── config/
│   └── orchestrator.yaml       # S3 paths, region, pipeline config
├── preprocess/
│   └── code/
│       ├── run.py              # Data loading, feature selection, train/test split
│       ├── config.py           # Preprocessing configuration loader
│       ├── config/
│       │   └── preprocess.yaml # Preprocessing-specific settings
│       ├── logger.py
│       └── requirements.txt    # SageMaker processing job dependencies
├── train/
│   └── code/
│       ├── train.py            # HuggingFace fine-tuning with evaluation metrics
│       └── logger.py
├── orchestrator.ipynb          # Full pipeline: preprocess → train → deploy
├── config.py                   # Root config loader (OmegaConf)
├── .env.example                # AWS credentials template
├── .github/
│   └── workflows/
│       └── ci-cd.yml           # GitHub Actions pipeline trigger
├── requirements.txt
└── README.md
```

## Tech Stack

| Category | Tools |
|----------|-------|
| **Orchestration** | AWS Step Functions |
| **ML Platform** | Amazon SageMaker (Processing, Training, Endpoints) |
| **Model** | HuggingFace DistilBERT (multi-label classification) |
| **Configuration** | OmegaConf + YAML |
| **CI/CD** | GitHub Actions |
| **Language** | Python 3.8+ |

## License

MIT License — see [LICENSE](LICENSE) for details.

## Author

**Soufiane Aazizi** — Lead AI Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/soufiane-aazizi-phd-a502829/)
[![Medium](https://img.shields.io/badge/Medium-000000?style=flat-square&logo=medium&logoColor=white)](https://medium.com/@aazizi.soufiane)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/aazizisoufiane)
