# 📂 Setup Instructions
## 1️⃣ Create/Activate Environment
### Using uv
Easiest way to setup the environment is to use [uv](https://github.com/astral-sh/uv).

1. Run `uv sync` to create the environment and install the project including its dependencies.
2. Run `source .env/bin/activate` to activate the environment.

### Using pip

1. Make sure you have python3.12 installed.
2. Create a virtual environment using `python3.12 -m venv .env`.
3. Activate the virtual environment using `source .env/bin/activate`.
4. Install the project and its dependencies using `pip install .`.

## 2️⃣ Prepare Dataset

1. Download the dataset from [Salicon Dataset Site](https://salicon.net/challenge-2017/) and select the "Download Fixation Maps". You need to Extract the zip and put the folder `train/` into `data/maps/` directory.
2. Run `python -m app.data data/maps data/filtered_data --correct-only` to create the filtered dataset.

## 🚀 Train the Model

Start the model training using:

```bash
python train.py +experiment=image_type
```

📌 **Now you're ready to process data and train your model! 🚀**


