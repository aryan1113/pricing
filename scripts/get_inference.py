'''
set for V1 right now, to use V3, please use train_V3
run all cells except the train loop
and load model from disk (price_predictor_V3.keras)
'''

import numpy as np
import os

import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import load_model

from scripts.utils import (
    CustomTokenizer,
    CNNImageEncoder,
    PricePredictor,
    prepare_inference_inputs
)

PROJECT_DIR = "./"

if __name__ == "__main__":

    title_vocab_path = os.path.join(PROJECT_DIR, "vocab/train_cleaned.csv_title")
    desc_vocab_path = os.path.join(PROJECT_DIR, "vocab/train_cleaned.csv_desc")
    model_path = os.path.join(PROJECT_DIR, "price_predictor_V1.keras") 

    title_max_tokens = 24
    desc_max_tokens = 48

    title_tokenizer = CustomTokenizer(title_vocab_path, title_max_tokens)
    desc_tokenizer = CustomTokenizer(desc_vocab_path, desc_max_tokens)

    try:
        model = load_model(model_path, custom_objects={'CNNImageEncoder': CNNImageEncoder, 'PricePredictor': PricePredictor})
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    sample_title = "Nintendo New 3DS"
    sample_desc = "Selling off this New 3DS (Non-XL). Comes with:- Console- Stylus- 32GB MicroSD- Charger Please Note: Device is modded, so can download any and all games from online. It also has a non-functioning volume slider, so the volume slider can't control the volume. Tried replacing the speakers and the cables but no luck in fixing this. However, because the console is modded the volume can still be adjusted anytime from the mod menu. Please purchase after careful consideration. Any questions feel free to ask, happy to answer any questions."
    sample_condition ="LIKE NEW"
    sample_image_url = "https://media.karousell.com/media/photos/products/2025/6/4/nintendo_new_3ds_1749051938_84b4b4d4_thumbnail"
    sample_date_sold = "2025-07-01"

    inference_inputs = prepare_inference_inputs(
        title=sample_title,
        description=sample_desc,
        image_url=sample_image_url,
        date_sold_str=sample_date_sold,
        condition_str=sample_condition,
        title_tokenizer=title_tokenizer,
        desc_tokenizer=desc_tokenizer
    )

    prediction_log_price = model.predict(inference_inputs)
    predicted_price = np.expm1(prediction_log_price[0][0])

    print(f"Predicted price for {sample_title} at {sample_condition}: ${predicted_price:.2f}")
