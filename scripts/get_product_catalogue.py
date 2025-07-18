import pandas as pd
import hashlib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    return ''.join(c.lower() if c.isalnum() or c.isspace() else ' ' for c in text).strip()


def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    required_cols = ['title', 'condition']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.dropna(subset=['title', 'condition'])
    df['title_clean'] = df['title'].apply(normalize_text)

    if 'Description' in df.columns:
        df['desc_clean'] = df['Description'].apply(normalize_text)
        df['title_clean'] = df.apply(
            lambda row: row['desc_clean'] if len(row['title_clean'].split()) < 4 else row['title_clean'], axis=1
        )

    return df


def extract_ngrams(df, ngram_range=(1, 3), min_df=5):
    '''
    get top n n-grams from title, 
    that occur in atleast min_df rows
    and at max in 80% of rows
    '''
    vectorizer = CountVectorizer(
        ngram_range=ngram_range, 
        min_df=min_df,
        max_df=0.8)
    X = vectorizer.fit_transform(df['title_clean'])
    vocab = vectorizer.get_feature_names_out()
    return vectorizer, vocab

def get_top_terms_per_title(vectorizer, df, top_n=3):
    X = vectorizer.transform(df['title_clean'])
    top_terms = []
    for row in X:
        term_scores = zip(row.indices, row.data)
        sorted_terms = sorted(term_scores, key=lambda x: -x[1])
        terms = [vectorizer.get_feature_names_out()[i] for i, _ in sorted_terms[:top_n]]
        top_terms.append(terms)
    df['TopTerms'] = top_terms
    return df


def generate_sku_id(attribute_list):
    key = '|'.join(attribute_list)
    # or maybe use somethnig simpler, like unique sku for each row 
    return hashlib.md5(key.encode()).hexdigest()[:10]


def group_to_skus(df):
    records = []
    for _, row in df.iterrows():
        terms = row['TopTerms']
        padded = terms + [''] * (3 - len(terms))
        condition = row['condition']
        key = padded + [condition]
        sku_id = generate_sku_id(key)

        record = {
            'SKU_ID': sku_id,
            'Attr_1': padded[0],
            'Attr_2': padded[1],
            'Attr_3': padded[2],
            'condition': condition,
            # skip for now, suprisingly most of this is nan
            # 'Price': row['Price'] if 'Price' in df.columns else np.nan
        }
        records.append(record)

    grouped_df = pd.DataFrame(records)
    agg_funcs = {
        'Attr_1': 'first',
        'Attr_2': 'first',
        'Attr_3': 'first',
        'condition': 'first',
        # 'Price': 'mean'
    }
    grouped = grouped_df.groupby('SKU_ID').agg(agg_funcs).reset_index()
    grouped['ListingCount'] = grouped_df.groupby('SKU_ID').size().values
    # grouped.rename(columns={'Price': 'AvgPrice'}, inplace=True)

    return grouped

def save_catalogue(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"Product Catalogue saved to {output_path} with {len(df)} SKU.")


def generate_product_catalogue(csv_path, output_path='product_catalogue.csv'):
    '''
    future work: 
    - how do I select ngram_range
    currently unigram, bigram and trigram taken into consideration

    - min_df 
    - minimum document frequency, to remove kess frequent n-gram

    '''
    df = load_and_preprocess(csv_path)
    vectorizer, vocab = extract_ngrams(df, ngram_range=(1, 3), min_df=3)
    df = get_top_terms_per_title(vectorizer, df, top_n=3)
    sku_df = group_to_skus(df)
    save_catalogue(sku_df, output_path)


if __name__ == '__main__':
    generate_product_catalogue('datasets/train.csv')


'''
Caveats

currently we rank n-gram based on frequency, not semantics
which results in rows like 003f9f1083,2ds,3ds,3ds 2ds,BRAND NEW,1
not very informative,
maybe can de-duplicate n-gram with predix matching

'''