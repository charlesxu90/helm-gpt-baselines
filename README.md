# HELMS-GPT baselines

## Installation and running
### Clone and Create environment
Clone git repository and then create the environment as follows.

```commandline
mamba env create -f environment.yml
conda activate helm-gpt-cmp-env
```

instal required packages
```commandline
pip install nltk joblib scikit-learn==0.23.2 numpy==1.21.4 loguru xgboost
pip install guacamol  # help solving many warnings
```

### Install dependencies to run agent

