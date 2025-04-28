import pandas as pd
import re
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertModel

# =========================
# Load Tokenizer and Model
# =========================
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Set model to evaluation mode
model.eval()

# =========================
# Clean Text Function
# =========================
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.strip()  # Remove leading/trailing whitespace
    return text

# =========================
# Prepare Embedding
# =========================
def prepare_embedding(text):
    # Clean the text
    text = clean_text(text)

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token

    return embeddings

# =========================
# Cosine Similarity
# =========================
def calculate_cosine_similarity(embedding1, embedding2):
    similarity = F.cosine_similarity(embedding1, embedding2)
    return similarity.item()

# =========================
# Process CSV
# =========================
def process_csv(csv_file):
    # Load CSV
    df = pd.read_csv(csv_file)

    # Display first few rows to confirm structure
    print("Sample Data:\n", df.head())
    print(df.columns)

    # Create a list to store similarity results
    results = []

    for index, row in df.iterrows():
        # if index >= 5:  # Process only first 5 rows
        #     break

        text1 = row['text1']
        text2 = row['text2']

        # Get embeddings
        embedding1 = prepare_embedding(text1)
        embedding2 = prepare_embedding(text2)

        # Calculate similarity
        similarity_score = calculate_cosine_similarity(embedding1, embedding2)

        results.append({
            'text1': text1,
            'text2': text2,
            'similarity_score': round(similarity_score, 4)  # Round for neatness
            
        })

    return results

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # Provide your CSV file path
    csv_file = r"/mnt/c/Users/jaish/Downloads/DataNeuron_DataScience_Task1/DataNeuron_Text_Similarity.csv"

    # Process the CSV and get similarity scores
    similarity_results = process_csv(csv_file)

    # Print some sample results
    for item in similarity_results[:5]:  # Print first 5 only
        print("\n---")
        print(f"Text 1: {item['text1']}")
        print(f"Text 2: {item['text2']}")
        print(f"Similarity Score: {item['similarity_score']}")
