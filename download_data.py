import os
import requests
import zipfile
import pandas as pd
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

# Define URLs and paths
url = "https://www.cs.jhu.edu/~mdredze/datasets/sentiment/domain_sentiment_data.zip"
download_path = "domain_sentiment_data.zip"
extract_path = "Multi-Domain_Sentiment_Dataset"
domains = ["books", "electronics", "kitchen"]  # Add other domains if needed

# Step 1: Download the dataset
if not os.path.exists(download_path):
    print("Downloading dataset...")
    response = requests.get(url)
    with open(download_path, "wb") as f:
        f.write(response.content)
    print("Download complete.")

# Step 2: Extract the dataset
if not os.path.exists(extract_path):
    print("Extracting dataset...")
    with zipfile.ZipFile(download_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction complete.")

# Initialize lists to store text and labels
texts, labels, domain_labels = [], [], []

# Step 3: Load and process data from each domain
for domain in domains:
    for label in ["positive", "negative"]:
        folder_path = os.path.join(extract_path, "domain_sentiment_data", domain, label)
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            continue
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read().strip())
                labels.append(1 if label == "positive" else 0)
                domain_labels.append(domain)

# Create DataFrame
multi_domain_df = pd.DataFrame({
    "Text": texts,
    "Sentiment": labels,
    "Domain": domain_labels
})

# Step 4: Split data into training and test sets
train_df, test_df = train_test_split(multi_domain_df, test_size=0.2, random_state=42, stratify=multi_domain_df["Domain"])

# Step 5: Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenization function
def preprocess_for_bert(data, tokenizer, max_length=128):
    """Tokenizes and preprocesses text for BERT."""
    tokens = tokenizer(
        data["Text"].tolist(),
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"  # Returns tensors for PyTorch
    )
    return tokens

# Apply tokenization to train and test sets
train_tokens = preprocess_for_bert(train_df, tokenizer)
test_tokens = preprocess_for_bert(test_df, tokenizer)

# Print a sample of tokenized inputs
print("Sample tokenized input for training set:")
print(train_tokens.input_ids[0])  # Display token IDs for first entry in the training set
print("Corresponding label:", train_df.iloc[0]["Sentiment"])

# Optional: Save tokenized data
train_df.to_csv("train_multi_domain.csv", index=False)
test_df.to_csv("test_multi_domain.csv", index=False)

print("Data processing complete. Ready for BERT model training.")
