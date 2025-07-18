# inference
Run scripts/inference.py, by specifying listing information in the file itself

## Notes (shared as pdf)
- [Google doc link](https://docs.google.com/document/d/1KJr9Cz5l2pC-5baJWJnXeLYeyu6ksFUfluhw9h_uOPo/edit?usp=sharing)
- [Results sheet](https://docs.google.com/spreadsheets/d/12mNKzSWa--2QllrEFT3lJVhQ-Lhfm7T5FCxTrcNoRkw/edit?usp=sharing)

## Repo structure

Using `tree -I '__pycache__|pricing-venv'`, can be simply installed using `brew install tree` (on macos)

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