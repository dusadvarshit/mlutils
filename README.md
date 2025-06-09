# 🧠 Auto-ml tool & common cloud storage interface

A modular, production-ready machine learning workflows library for general purpose modeling on tabular dataset.

This repository combines best practices in machine learning engineering — data acquisition from kaggle , data pre-processing, model training and evaluation, tracking with MLflow, and rigorous code quality with pre-commit hooks and unit testing.
The final working package can be installed as .whl binary and can be built with simple commands after cloning this github repo.

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
- ✅ Integration with top 3 cloud storage services
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
    * A backend store either locally or remotely needs to be created. Same thing applies to a backend db as well.
    * Postgres db connection string needs to be provided
    * For using another backend store natively compatible with mlflow, please pass their db connection string and backend store url
    * In case you want to run the local mlflow server, uncomment the commands following the comment starting with "For local testing..." and comment the current commands.

* To use the respective cloud services their respective credentials need to be passed to the working environment.
    * For local uses, a simple .env file inside the current repo is sufficient. The gitignore config already excludes .env files thus voiding the chances of accidentally committing confidential secrets.
    * For additional security, the ENV variables can be set globally directly from terminal or a bash script outside the repo.
    * For production use cases, it is best to use a managed service such as AWS KMS/Hashicorp Vault or equivalent to pass env variables.
    * For simpler production cases such as an isolated service deployed on ECS one can specify ENV var directly in task specification or link to a file. (This is AWS specific, follow other vendors' docs for their specific steps)
    * ENV VARS that can be specified. None of them are a must unless the specific service is meant to be used.
        * KAGGLE_KEY
        * KAGGLE_USERNAME
        * KAGGLEHUB_CACHE
        * AZ_TENANT_ID
        * AZ_CLIENT_ID
        * AZ_CLIENT_SECRET
        * AZ_STORAGE_ACCOUNT_NAME
        * AZ_CONTAINER_NAME
        * GCP_SERVICE_ACCOUNT_JSON
        * AWS_ACCESS_KEY_ID
        * AWS_SECRET_ACCESS_KEY
        * AWS_DEFAULT_REGION
        * POSTGRES_DB

## 1. Package install
```
git clone https://github.com/mandrake-bio/mlutils
cd mlutils

## For local testing
poetry install

## For distributing production build
poetry build
pip install dist/mlutils-0.1.0-py3-none-any.whl

./mlflow.sh

```

## 2. How to use?
Please follow the jupyter notebook titled Tutorial.ipynb for detailed explanation on how to use the package.

## 3. Running Tests 🧪
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

Pre-commit hooks for linting, syntax errors, formatting, and secrets

MLflow for reproducible ML experiments

Logging to track events

Unit tests for all major components.

# 📁 Notebooks & Data
Jupyter notebooks for data exploration and EDA are in the /notebooks directory.

Training and test datasets should be placed in relevant subdirectories inside /data.

# Future Directions
* Handle file formats outside of csv
* Handle dataset comprised of multiple files
* Support for multi-label classification
    * Updating param_grid models to pass multi-label hyper-params
* Allow for custom pre-processing and feature engineering
* Support for neural networks
    * Add support for both pytorch and tensorflow
    * Integrate tf/pytorch nns directly with sklearn pipeline for tabular datasets
    * Create new interface - for unstructured data: images, audio, natural language.
* Cloud
    * Unifying cloud interface with a single class
    * Adding support for functionality beyond just storage:
        * Interacting with LLMs
        * Interacting with databases
        * Analytics on cloud compute usage
* MLFlow enhancements
    * Incorporating SHAP plots into mlflow artifacts
    * Automatically register model based on experiment metrics
    * Automatically serve model
* Enhancing unit test coverage

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
