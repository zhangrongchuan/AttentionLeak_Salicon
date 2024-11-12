# Setup

## Using uv
Easiest way to setup the environment is to use [uv](https://github.com/astral-sh/uv).

1. Run `uv sync` to create the environment and install the project including its dependencies.
2. Run `source .env/bin/activate` to activate the environment.

## Using pip

1. Make sure you have python3.12 installed.
2. Create a virtual environment using `python3.12 -m venv .env`.
3. Activate the virtual environment using `source .env/bin/activate`.
4. Install the project and its dependencies using `pip install .`.

## Prepare Dataset

1. Download the dataset `SalChartQA.zip` from [SalChartQA Project Site](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-3884) and extract the zip into `data/` directory.
2. Run `python -m app.data data/SalChartQA/ data/filtered_question_type_ans_correct --correct-only` to create the filtered dataset.

## Train

1. Run `python train.py +experiment=chart_simple` to train the model on the attribute `chart_simple`.
