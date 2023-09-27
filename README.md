# CI/CD MLOps with SageMaker and Step Functions
![Author](https://img.shields.io/badge/Author-Soufiane%20AAZIZI-brightgreen)
[![Medium](https://img.shields.io/badge/Medium-Follow%20Me-blue)](https://medium.com/@aazizi.soufiane)
[![GitHub](https://img.shields.io/badge/GitHub-Follow%20Me-lightgrey)](https://github.com/aazizisoufiane)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect%20with%20Me-informational)](https://www.linkedin.com/in/soufiane-aazizi-phd-a502829/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project serves as an introductory example of how to set up a CI/CD pipeline for Machine Learning Operations (MLOps)
using Amazon SageMaker and AWS Step Functions. It's a valuable resource for those looking to streamline their ML
workflows and automate the deployment of models.

Feel free to explore the code and adapt it to your specific ML use cases. If you have any questions or need further
assistance, please don't hesitate to reach out.

## Project Structure

The project structure is organized as follows:
```plaintext
├── .github
│   └── workflows
│       └── ci-cd.yml
├── README.md
├── config
│   └── orchestrator.yaml
├── config.py
├── orchestrator.ipynb
├── preprocess
│   └── code
│       ├── config
│       │   └── preprocess.yaml
│       ├── config.py
│       ├── logger.py
│       ├── requirements.txt
│       └── run.py
├── requirements.txt
└── train
    └── code
        ├── logger.py
        └── train.py
```


- `.github/workflows`: Contains the GitHub Actions workflow configuration for Continuous Integration/Continuous
  Deployment (CI/CD).
    - `ci-cd.yml`: Defines the CI/CD pipeline that automates the preprocessing, training, and deployment tasks.

- `config`: Configuration files for the project.
    - `orchestrator.yaml`: Configuration file for the Step Functions state machine.

- `config.py`: Python module for project-specific configurations.

- `orchestrator.ipynb`: Jupyter Notebook demonstrating the entire workflow using SageMaker and Step Functions. Convert this script to `orchestrator.py` to use the CI/CD feature.

- `preprocess`: Directory containing preprocessing related code.
    - `code`: Code for data preprocessing.
        - `config`: Configuration files for preprocessing.
        - `config.py`: Python module for preprocessing configurations.
        - `logger.py`: Logging utility for preprocessing.
        - `requirements.txt`: List of Python dependencies for preprocessing.
        - `run.py`: Script for running data preprocessing.

- `requirements.txt`: List of Python dependencies for the entire project.

- `train`: Directory containing training related code.
    - `code`: Code for model training.
        - `logger.py`: Logging utility for training.
        - `train.py`: Script for training the machine learning model.

## Getting Started

Follow these steps to get started with the project:

1. **Clone the Repository**: Clone this GitHub repository to your local development environment.

   ```bash
   git clone https://github.com/your-username/your-repo.git
   ```

2. **Install Dependencies**: Install the Python dependencies required for preprocessing and training.
   
    ```bash 
    pip install -r requirements.txt 
    ```
3. **Configure AWS Credentials**: Ensure you have configured your AWS credentials and have the necessary permissions to use SageMaker and Step Functions.

4. **Run the Workflow**: Open and run the orchestrator.ipynb Jupyter Notebook to execute the complete CI/CD workflow. This notebook will guide you through each step of the process.

## CI/CD Pipeline
The CI/CD pipeline defined in .github/workflows/ci-cd.yml automates the entire machine learning workflow, including data preprocessing, model training, and deployment. When code is pushed or tagged, the pipeline automatically triggers the workflow to keep your ML model up to date.
        
  
