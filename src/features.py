import pandas as pd
import re

# List of suspicious words to flag in reviews
sus_words = ["free", "amazing", "best", "buy now", "limited", "guaranteed"]

def count_sus_words(reviews: pd.DataFrame, suspicious_list: list) -> pd.DataFrame:
    """
    Count the number of suspicious words in each review based on a predefined list.

    Parameters:
    - reviews (pd.DataFrame): DataFrame containing a 'text_' column with review texts.
    - suspicious_list (list): List of suspicious keywords to search for.

    Returns:
    - pd.DataFrame: Original DataFrame with an additional 'count_sus' column.
    """
    counts = []
    for text in reviews['text_']:
        text_lower = text.lower()
        count = sum(text_lower.count(word) for word in suspicious_list)
        counts.append(count)
    reviews['count_sus'] = counts
    return reviews

def count_capital_words(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Count the number of fully capitalized words in each review.

    Parameters:
    - reviews (pd.DataFrame): DataFrame containing a 'text_' column with review texts.

    Returns:
    - pd.DataFrame: Original DataFrame with an additional 'count_caps' column.
    """
    capital_counts = []
    for text in reviews['text_']:
        count = len([word for word in text.split() if word.isupper()])
        capital_counts.append(count)
    reviews['count_caps'] = capital_counts
    return reviews

def count_review_length(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the length of each review in terms of number of words.

    Parameters:
    - reviews (pd.DataFrame): DataFrame containing a 'text_' column with review texts.

    Returns:
    - pd.DataFrame: Original DataFrame with an additional 'len_reviews' column.
    """
    lengths = [len(text.split()) for text in reviews['text_']]
    reviews['len_reviews'] = lengths
    return reviews