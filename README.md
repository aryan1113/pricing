## How to use this ?
Download locally, simple git clone <br>
Then 
- create a venv using python3.11 `python3.11 -m venv check`, 
- activate it using `source check/bin/activate`,
- install requirements using `pip install -r requirements.txt`

To get inference, run `python -m scripts.get_inference`

## Moving around this repo
1. /datasets, for csv inputs
2. /scripts, for pricing inference, generating product catalogue, saving logs to csv and utils to keep get_inference.py simple
3. /submissions, well the heart of it all
4. /train, two models V1 and V3
5. /vocab, separate for title and description, should be re-generated for other categories (othan than gaming consoles)
6. eda.ipynb , simple charts to just understand the data
7. price_predictor_*.keras, model save to disk, loaded during inference

## Notes (shared as pdf)
- [Google doc link](https://docs.google.com/document/d/1KJr9Cz5l2pC-5baJWJnXeLYeyu6ksFUfluhw9h_uOPo/edit?usp=sharing)
- [Results sheet](https://docs.google.com/spreadsheets/d/12mNKzSWa--2QllrEFT3lJVhQ-Lhfm7T5FCxTrcNoRkw/edit?usp=sharing)

## How to use the predictor / inference ?
Simply run python -m run `scripts.get_inference.py`, inputs have been specified in the file itself, <br>
pass in values     
- sample_title 
- sample_desc 
- sample_condition ="LIKE NEW"
- sample_image_url = "https://media.karousell.com/media/photos/products/2025/6/4/nintendo_new_3ds_1749051938_84b4b4d4_thumbnail"
- sample_date_sold = "2025-07-01"

## To generate the product catalogue for any category
Based on n-grams
Run scripts/get_product_catalogue.py , just update the call which is currently set as `generate_product_catalogue('datasets/train.csv')` <br>
(only update train.csv => desired input file)

## Repo structure

<pre lang="markdown"> <code>Using `tree -I '__pycache__|pricing-venv'`, can be simply installed using `brew install tree` (on macos)

├── eda.ipynb
├── initial_thoughts.txt
├── price_predictor_V1.keras
├── price_predictor_V3.keras
├── requirements.txt
│
├── scripts
│   ├── get_inference.py
│   ├── get_product_catalogue.py
│   ├── save_logs.py
│   └── utils.py
│
├── submissions
│   ├── product_catalogue.csv
│   └── submission_a.csv
│
├── train 
│       train_* jupyter notebooks
│       predicted_prices_* respective predictions
│       training_log_* train loss, validation loss
│
│   ├── predicted_prices_v1.csv
│   ├── predicted_prices_v3.csv
│   ├── train_v1.ipynb
│   ├── train_v3.ipynb
│   ├── training_log_V1.csv
│   └── training_log_V3.csv
│
└── vocab 
    ├── train_cleaned.csv_desc
    └── train_cleaned.csv_title
</code> </pre>
