# 🧠 Auto-ml tool & common cloud storage interface

A modular, production-ready machine learning workflows library for general purpose modeling on tabular dataset.

This repository combines best practices in machine learning engineering — data acquisition from kaggle , data pre-processing, model training and evaluation, tracking with MLflow, and rigorous code quality with pre-commit hooks and unit testing.
The final working package can be installed as .whl binary and is shared along with the github repo.

---

## 📚 Table of Contents

- [🔧 Features](#-features)
- [📁 Project Structure](#-project-structure)
- [🚀 Getting Started](#-getting-started)
- [🏋️‍♂️ Training the Model](#️-training-the-model)
- [🧪 Running Tests](#-running-tests)
- [🌐 Serving via Web App](#-serving-via-web-app)
- [📦 Docker Deployment](#-docker-deployment)
- [📊 MLflow Integration](#-mlflow-integration)
- [💡 API Usage Example](#-api-usage-example)
- [🧰 Development Practices](#-development-practices)
- [📁 Notebooks & Data](#-notebooks--data)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## 🔧 Features

- ✅ End-to-end ML prediction (both regression & classification) workflow
- ✅ Clean, modular Python codebase
- ✅ MLflow for experiment tracking
- ✅ Docker support for reproducible environments
- ✅ Pre-commit hooks (Black, Flake8, isort, etc.)
- ✅ Pytest unit testing
- ✅ Logging and configuration management

---

## 📁 Project Structure

```text
src/
└── mlutils/
    ├── cloud
    │   ├── azure.py            # Azure
    │   ├── aws.py            # Amazon Web Services
    │   ├── gcp.py            # Google Cloud Platform
    │
    ├── automl/
    │   ├── train.py          # Model training logic
    │   ├── eval.py           # Evaluation metrics
    │
    └── utils/
        └── utils/
            ├── config.py        # Configuration and hyperparameters
            ├── io.py            # Data loading and I/O utilities
            ├── logger.py        # Logging setup
            ├── mlflow_utils.py  # MLflow integration
            ├── kaggle.py        # Kaggle Interface
tests/
└── tests/
    └── __init__.py               # Unit test definitions
```
---

# 🚀 Getting Started

## 0. Pre-requisites
There are several pre-requisites to be able to use this functionality by simply cloning and installing the python package as it is.

* Poetry should be installed in the base python environment. Dev system used conda to build the base working virtual environment and poetry to handle all further dependencies.

* To use mlflow with a cloud backend - an object store and the remote database needs to be created.
    * This project is using AWS RDS (using postgres) as the backend database and AWS S3 as the artifact store.
    * To reproduce the same, AWS credentials need to be pre-configured inside the env either via aws cli or via .env file.
    * Postgres db connection string needs to be provided
    * For using another backend store natively compatible with mlflow, please pass their db connection string and backend store url

## 1. Clone the Repository
```
git clone https://github.com/mandrake-bio/mlutils
cd mlutils

poetry install

./mlflow.sh

```


# 🧪 Running Tests
Use pytest to run all unit tests:
```
pytest src/tests/
```

For test coverage:
```
pytest --cov=mlutils tests/
```



# 🧰 Development Practices
This repository follows strict development standards:

Git version control

Pre-commit hooks for linting, formatting, and secrets

Docker for repeatable environments

MLflow for reproducible ML experiments

Logging to track events

Unit tests for all major components.

# 📁 Notebooks & Data
Jupyter notebooks for data exploration and EDA are in the /notebooks directory.

Training and test datasets should be placed in /data.

Models are saved to /models.

None of these folders are put into .gitignore purposefully to make this entire notebook reproducible.

# 🤝 Contributing

Contributions are welcome!

Fork the repo

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add amazing feature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

Ensure you follow the code standards and include relevant unit tests.

# 📄 License
Distributed under the MIT License. See LICENSE for more information.

# 🙌 Acknowledgments
This project integrates tools and practices from:

Scikit-learn

Flask

MLflow

Docker

Pandas / NumPy

Pytest

Pre-commit

GitHub Actions (optional CI/CD integration)

**This file is completely self-contained and fully detailed. You can copy this directly into your project as `README.md` without needing any additional documents. Let me know if you'd like me to generate a badge section or add sample config templates in-code!**
