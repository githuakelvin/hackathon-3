import nltk
import ssl

# Fix SSL certificate issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download all required NLTK data
datasets = [
    'punkt',           # For tokenization
    'punkt_tab',       # Additional tokenizer data
    'stopwords',       # Common stopwords
    'wordnet',         # WordNet lemmatizer
    'averaged_perceptron_tagger'  # For better tokenization
]

print("Downloading NLTK datasets...")
for dataset in datasets:
    try:
        nltk.download(dataset, quiet=False)
        print(f"✓ Successfully downloaded {dataset}")
    except Exception as e:
        print(f"✗ Error downloading {dataset}: {e}")

print("All NLTK datasets downloaded!")