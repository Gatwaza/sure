import pandas as pd
from transformers import AutoTokenizer

def preprocess_data(dataset_path):
    # Load the dataset
    df = pd.read_csv(dataset_path)
    
    # Ensure the 'Heart Disease' column is used as text
    if 'Heart Disease' not in df.columns:
        raise KeyError("The dataset must contain a 'Heart Disease' column.")
    
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Tokenize texts
    texts = df['Heart Disease'].tolist()
    tokenized_texts = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    
    # Extract labels (assuming binary classification with 'Presence' and 'Absence')
    labels = df['Heart Disease'].apply(lambda x: 1 if x == 'Presence' else 0).tolist()
    
    return tokenized_texts, labels
