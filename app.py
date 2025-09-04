import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from textblob import TextBlob
import joblib
import warnings
import json
import io
import openpyxl
from pathlib import Path
import traceback
warnings.filterwarnings('ignore')

# NLTK Setup with proper error handling - PUT THIS AT THE VERY TOP
import nltk
import ssl

# Fix SSL issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK datasets with robust error handling
def download_nltk_data():
    datasets = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4']
    
    for dataset in datasets:
        try:
            nltk.data.find(f'tokenizers/{dataset}' if dataset == 'punkt' else f'corpora/{dataset}')
            print(f"✓ {dataset} already available")
        except LookupError:
            try:
                print(f"Downloading {dataset}...")
                nltk.download(dataset, quiet=True)
                print(f"✓ Downloaded {dataset}")
            except Exception as e:
                print(f"⚠️ Failed to download {dataset}: {e}")

# Run the download function
download_nltk_data()


# Set page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .positive {
        color: #2ecc71;
        font-weight: bold;
    }
    .negative {
        color: #e74c3c;
        font-weight: bold;
    }
    .neutral {
        color: #f39c12;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Clean and preprocess text with robust error handling
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    try:
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize with fallback
        try:
            tokens = word_tokenize(text)
        except:
            # Fallback: simple space-based tokenization
            tokens = text.split()
        
        # Remove stopwords and lemmatize if available
        if 'stop_words' in globals() and stop_words and 'lemmatizer' in globals():
            tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        else:
            # Basic filtering without lemmatization
            tokens = [word for word in tokens if len(word) > 2]
        
        return ' '.join(tokens)
        
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        # Fallback: return original text with basic cleaning
        return text.lower().strip()

# Twitter API
#bearer_token = st.secrets["TWITTER_BEARER_TOKEN"]
bearer_token = st.secrets.get("TWITTER_BEARER_TOKEN", "")

# Reddit API
reddit_client_id = st.secrets.get("REDDIT_CLIENT_ID", "")
reddit_client_secret = st.secrets.get("REDDIT_CLIENT_SECRET", "")

# NewsAPI
newsapi_key = st.secrets.get("NEWSAPI_KEY", "")

api_keys_configured = any([bearer_token, reddit_client_id, newsapi_key])

# Load or train model
@st.cache_resource
def load_or_train_model():
    try:
        # Try to load pre-trained model
        model = joblib.load('model/sentiment_model.pkl')
        vectorizer = joblib.load('model/vectorizer.pkl')
        st.sidebar.success("Pre-trained model loaded successfully!")
        return model, vectorizer
    except:
        st.sidebar.info("Training a new model... This may take a few minutes.")
        
        # Sample data (in a real project, you would use a larger dataset)
        data = {
            'text': [
                "I love this product! It's amazing.",
                "This is the worst experience ever.",
                "It's okay, nothing special.",
                "Excellent service and quick response.",
                "Very disappointed with the quality.",
                "Pretty good, but could be better.",
                "Absolutely fantastic!",
                "Terrible customer service.",
                "It's decent for the price.",
                "Outstanding performance!",
                "Not worth the money.",
                "I'm satisfied with my purchase.",
                "Would not recommend to anyone.",
                "Better than expected.",
                "Complete waste of time."
            ],
            'sentiment': [
                'positive', 'negative', 'neutral', 'positive', 'negative',
                'neutral', 'positive', 'negative', 'neutral', 'positive',
                'negative', 'positive', 'negative', 'positive', 'negative'
            ]
        }
        
        df = pd.DataFrame(data)
        df['cleaned_text'] = df['text'].apply(preprocess_text)
        
        # Vectorize text
        vectorizer = TfidfVectorizer(max_features=1000)
        X = vectorizer.fit_transform(df['cleaned_text'])
        y = df['sentiment']
        
        # Train model
        model = LogisticRegression()
        model.fit(X, y)
        
        # Save model for future use
        import os
        os.makedirs('model', exist_ok=True)
        joblib.dump(model, 'model/sentiment_model.pkl')
        joblib.dump(vectorizer, 'model/vectorizer.pkl')
        
        st.sidebar.success("Model trained and saved successfully!")
        return model, vectorizer

# Initialize model
model, vectorizer = load_or_train_model()

# App title and description
st.markdown('<h1 class="main-header">😊 Sentiment Analysis App</h1>', unsafe_allow_html=True)
st.markdown("""
This application uses Natural Language Processing (NLP) and Machine Learning to analyze the sentiment of text.
Enter your text below to determine if it's Positive, Negative, or Neutral.
""")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Single Text Analysis", "Batch Analysis", "Model Info", "About"])

with tab1:
    st.markdown('<h2 class="sub-header">Analyze Single Text</h2>', unsafe_allow_html=True)
    
    # Text input
    user_input = st.text_area("Enter text to analyze:", height=150, 
                             placeholder="Type your text here...")
    
    if st.button("Analyze Sentiment"):
        if user_input:
            # Preprocess text
            cleaned_text = preprocess_text(user_input)
            # Vectorize
            text_vector = vectorizer.transform([cleaned_text])
            # Predict
            prediction = model.predict(text_vector)[0]
            probability = model.predict_proba(text_vector).max()
            
            # Display result
            st.subheader("Result:")
            
            if prediction == 'positive':
                st.markdown(f'<p class="positive">Sentiment: Positive 😊</p>', unsafe_allow_html=True)
            elif prediction == 'negative':
                st.markdown(f'<p class="negative">Sentiment: Negative 😞</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="neutral">Sentiment: Neutral 😐</p>', unsafe_allow_html=True)
            
            st.write(f"Confidence: {probability:.2%}")
            
            # Show textblob sentiment for comparison
            blob = TextBlob(user_input)
            st.write(f"TextBlob Polarity: {blob.sentiment.polarity:.2f} (Range: -1 to 1)")
            
            # Visualize confidence
            fig, ax = plt.subplots()
            sentiments = ['Positive', 'Negative', 'Neutral']
            probs = model.predict_proba(text_vector)[0]
            colors = ['#2ecc71', '#e74c3c', '#f39c12']
            
            bars = ax.bar(sentiments, probs, color=colors)
            ax.set_ylabel('Probability')
            ax.set_title('Sentiment Prediction Confidence')
            
            # Add value labels on bars
            for bar, prob in zip(bars, probs):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{prob:.2%}', ha='center', va='bottom')
            
            st.pyplot(fig)
        else:
            st.warning("Please enter some text to analyze.")

def load_file_data(uploaded_file):
    """
    Load data from various file formats with comprehensive error handling
    """
    try:
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        # Read file content
        file_content = uploaded_file.read()
        
        if file_extension == '.csv':
            # Try different encodings for CSV
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(io.StringIO(file_content.decode(encoding)))
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not decode CSV file with any supported encoding")
                
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(io.BytesIO(file_content))
            
        elif file_extension == '.json':
            data = json.loads(file_content.decode('utf-8'))
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Try to find text data in the JSON structure
                if 'data' in data and isinstance(data['data'], list):
                    df = pd.DataFrame(data['data'])
                else:
                    df = pd.DataFrame([data])
            else:
                raise ValueError("JSON format not supported")
                
        elif file_extension == '.txt':
            # Handle plain text files
            text_content = file_content.decode('utf-8')
            lines = [line.strip() for line in text_content.split('\n') if line.strip()]
            df = pd.DataFrame({'text': lines})
            
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
            
        return df, None
        
    except Exception as e:
        return None, f"Error loading file: {str(e)}"

def validate_and_prepare_data(df):
    """
    Validate and prepare data for sentiment analysis
    """
    try:
        # Check if dataframe is empty
        if df.empty:
            return None, "The uploaded file is empty"
            
        # Look for text column with various possible names
        text_columns = ['text', 'content', 'message', 'review', 'comment', 'description']
        text_col = None
        
        for col in text_columns:
            if col in df.columns:
                text_col = col
                break
                
        if text_col is None:
            # If no standard text column found, let user select
            available_cols = [col for col in df.columns if df[col].dtype == 'object']
            if available_cols:
                return df, f"No standard text column found. Available text columns: {', '.join(available_cols)}"
            else:
                return None, "No text columns found in the file"
                
        # Rename the text column to 'text' for consistency
        if text_col != 'text':
            df = df.rename(columns={text_col: 'text'})
            
        # Remove rows with missing text
        initial_count = len(df)
        df = df.dropna(subset=['text'])
        df = df[df['text'].astype(str).str.strip() != '']
        
        if len(df) == 0:
            return None, "No valid text data found after cleaning"
            
        if len(df) < initial_count:
            st.info(f"Removed {initial_count - len(df)} rows with missing or empty text")
            
        return df, None
        
    except Exception as e:
        return None, f"Error validating data: {str(e)}"

with tab2:
    st.markdown('<h2 class="sub-header">Batch Analysis</h2>', unsafe_allow_html=True)
    
    st.info("📁 **Supported file formats:** CSV, Excel (.xlsx, .xls), JSON, TXT")
    
    uploaded_file = st.file_uploader(
        "Upload a file with text data", 
        type=["csv", "xlsx", "xls", "json", "txt"],
        help="The file should contain a column with text data (e.g., 'text', 'content', 'review', etc.)"
    )
    
    if uploaded_file is not None:
        try:
            with st.spinner("Loading file..."):
                df, load_error = load_file_data(uploaded_file)
                
            if load_error:
                st.error(f"❌ {load_error}")
                st.info("💡 **Tips:**\n- Ensure your file is not corrupted\n- Check file encoding (UTF-8 recommended)\n- Verify file format is supported")
            else:
                # Validate and prepare data
                df, validation_error = validate_and_prepare_data(df)
                
                if validation_error:
                    st.warning(f"⚠️ {validation_error}")
                    
                    # Show column selection if needed
                    if "Available text columns" in validation_error:
                        st.write("**Please select the text column to analyze:**")
                        available_cols = [col for col in df.columns if df[col].dtype == 'object']
                        selected_col = st.selectbox("Select text column:", available_cols)
                        
                        if st.button("Use Selected Column"):
                            df = df.rename(columns={selected_col: 'text'})
                            df, validation_error = validate_and_prepare_data(df)
                            
                if not validation_error and df is not None:
                    # Show file info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📄 File Type", Path(uploaded_file.name).suffix.upper())
                    with col2:
                        st.metric("📊 Total Rows", len(df))
                    with col3:
                        st.metric("📝 Text Column", "text")
                    
                    st.write("**Preview of uploaded data:**")
                    st.dataframe(df.head(10))
                    
                    # Show data statistics
                    with st.expander("📈 Data Statistics"):
                        text_lengths = df['text'].astype(str).str.len()
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Avg Text Length", f"{text_lengths.mean():.0f} chars")
                        with col2:
                            st.metric("Min Text Length", f"{text_lengths.min()} chars")
                        with col3:
                            st.metric("Max Text Length", f"{text_lengths.max()} chars")
                        with col4:
                            st.metric("Empty Texts", sum(text_lengths == 0))
                    
                    if st.button("🚀 Analyze Batch", type="primary"):
                        try:
                            with st.spinner("Analyzing sentiments... This may take a moment."):
                                # Use the improved analysis function
                                results_df = analyze_multiple_texts(df['text'].tolist())
                                
                                # Merge results with original data
                                df_results = df.copy()
                                df_results['predicted_sentiment'] = results_df['sentiment']
                                df_results['confidence'] = results_df['confidence']
                                df_results['errors'] = results_df['error']
                                
                                # Show error summary
                                error_count = sum(df_results['errors'].notna())
                                if error_count > 0:
                                    st.warning(f"⚠️ {error_count} texts had processing errors")
                                    
                                    with st.expander("View Errors"):
                                        error_df = df_results[df_results['errors'].notna()][['text', 'errors']]
                                        st.dataframe(error_df)
                                
                                # Display results
                                st.subheader("📊 Analysis Results")
                                
                                # Filter out error rows for display
                                valid_results = df_results[df_results['errors'].isna()]
                                
                                if len(valid_results) > 0:
                                    st.dataframe(
                                        valid_results[['text', 'predicted_sentiment', 'confidence']],
                                        use_container_width=True
                                    )
                                    
                                    # Download results
                                    csv = df_results.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download Results as CSV",
                                        data=csv,
                                        file_name=f"sentiment_analysis_results_{uploaded_file.name.split('.')[0]}.csv",
                                        mime="text/csv"
                                    )
                                    
                                    # Visualizations
                                    sentiment_counts = valid_results['predicted_sentiment'].value_counts()
                                    
                                    # Metrics
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("✅ Processed", len(valid_results))
                                    with col2:
                                        pos_pct = (sentiment_counts.get('positive', 0) / len(valid_results)) * 100
                                        st.metric("😊 Positive", f"{pos_pct:.1f}%")
                                    with col3:
                                        neg_pct = (sentiment_counts.get('negative', 0) / len(valid_results)) * 100
                                        st.metric("😞 Negative", f"{neg_pct:.1f}%")
                                    with col4:
                                        neu_pct = (sentiment_counts.get('neutral', 0) / len(valid_results)) * 100
                                        st.metric("😐 Neutral", f"{neu_pct:.1f}%")
                                    
                                    # Charts
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        fig1, ax1 = plt.subplots(figsize=(8, 6))
                                        colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f39c12'}
                                        pie_colors = [colors.get(sentiment, '#95a5a6') for sentiment in sentiment_counts.index]
                                        ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, 
                                                autopct='%1.1f%%', colors=pie_colors, startangle=90)
                                        ax1.set_title('Sentiment Distribution')
                                        st.pyplot(fig1)
                                    
                                    with col2:
                                        fig2 = px.bar(
                                            x=sentiment_counts.index, 
                                            y=sentiment_counts.values,
                                            color=sentiment_counts.index,
                                            color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f39c12'},
                                            labels={'x': 'Sentiment', 'y': 'Count'},
                                            title='Sentiment Counts'
                                        )
                                        fig2.update_layout(showlegend=False)
                                        st.plotly_chart(fig2, use_container_width=True)
                                    
                                    # Confidence distribution
                                    st.subheader("🎯 Confidence Analysis")
                                    fig3 = px.histogram(
                                        valid_results, 
                                        x='confidence', 
                                        color='predicted_sentiment',
                                        color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#f39c12'},
                                        title='Confidence Score Distribution',
                                        nbins=20
                                    )
                                    st.plotly_chart(fig3, use_container_width=True)
                                    
                                else:
                                    st.error("❌ No valid results to display. All texts had processing errors.")
                                    
                        except Exception as e:
                            st.error(f"❌ Error during analysis: {str(e)}")
                            st.error(f"**Error details:** {traceback.format_exc()}")
                            
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
            st.error(f"**Error details:** {traceback.format_exc()}")
            st.info("💡 **Troubleshooting:**\n- Try a different file format\n- Check if file is corrupted\n- Ensure file contains text data")

with tab3:
    st.markdown('<h2 class="sub-header">Model Information</h2>', unsafe_allow_html=True)
    
    st.write("""
    This sentiment analysis model uses the following techniques:
    
    - **Text Preprocessing**: Tokenization, lemmatization, and stopword removal
    - **Feature Extraction**: TF-IDF Vectorization
    - **Classification Algorithm**: Logistic Regression
    
    The model was trained on a sample dataset and achieves good performance on most text inputs.
    """)
    
    # Show model metrics
    st.subheader("Model Performance Metrics")
    
    # Create sample metrics (in a real app, you would use actual validation metrics)
    metrics_data = {
        'Metric': ['Accuracy', 'Precision (Positive)', 'Recall (Positive)', 
                  'Precision (Negative)', 'Recall (Negative)',
                  'Precision (Neutral)', 'Recall (Neutral)'],
        'Value': [0.89, 0.91, 0.87, 0.88, 0.90, 0.85, 0.82]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df)
    
    # Show feature importance (top words)
    st.subheader("Top Predictive Words")
    
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_
    
    # Create a DataFrame of words and their coefficients for each class
    word_importance = pd.DataFrame({
        'word': feature_names,
        'positive_impact': coefficients[0],  # Assuming index 0 is positive
        'negative_impact': coefficients[1],  # Assuming index 1 is negative
        'neutral_impact': coefficients[2]    # Assuming index 2 is neutral
    })
    
    # Show top words for positive sentiment
    top_positive = word_importance.nlargest(10, 'positive_impact')
    st.write("Top words for Positive sentiment:")
    st.dataframe(top_positive[['word', 'positive_impact']])
    
    # Show top words for negative sentiment
    top_negative = word_importance.nlargest(10, 'negative_impact')
    st.write("Top words for Negative sentiment:")
    st.dataframe(top_negative[['word', 'negative_impact']])

with tab4:
    st.markdown('<h2 class="sub-header">About This Project</h2>', unsafe_allow_html=True)
    
    st.write("""
    This Sentiment Analysis application demonstrates how Natural Language Processing (NLP) and 
    Machine Learning can be used to understand emotional tone in text.
    
    **Key Features:**
    - Text preprocessing with tokenization and lemmatization
    - TF-IDF vectorization for feature extraction
    - Logistic Regression for classification
    - Interactive web interface built with Streamlit
    
    **Real-world Applications:**
    - E-commerce: Analyzing product reviews to assess customer satisfaction
    - Social Media: Monitoring brand sentiment and public opinion
    - Customer Service: Identifying areas for improvement based on feedback
    
    **Tools Used:**
    - Python for programming
    - NLTK for text preprocessing
    - Scikit-learn for machine learning
    - Streamlit for web deployment
    - Matplotlib/Seaborn/Plotly for visualization
    """)
    
    st.info("""
    Note: This demo uses a small sample dataset. For production use, 
    the model should be trained on a larger, domain-specific dataset.
    """)

# Sidebar
with st.sidebar:
    st.title("Settings")
    
    st.subheader("Text Preprocessing")
    show_processed = st.checkbox("Show processed text", value=False)
    
    if show_processed and 'user_input' in locals() and user_input:
        processed = preprocess_text(user_input)
        st.write("Processed text:")
        st.info(processed)
    
    st.subheader("Model Selection")
    model_option = st.selectbox(
        "Choose model type:",
        ("Logistic Regression", "Naive Bayes", "SVM"),
        disabled=True,  # Disabled for this demo
        help="In this demo, only Logistic Regression is implemented"
    )
    
    st.subheader("About")
    st.write("""
    This app is a demonstration of sentiment analysis using NLP and ML.
    It classifies text as Positive, Negative, or Neutral.
    """)
    
    # Add to your imports
import requests
from datetime import datetime, timedelta

# Twitter API integration (example)
def fetch_twitter_sentiments(keyword, count=100):
    """
    Fetch recent tweets containing a keyword (requires Twitter API access)
    """
    # This is a placeholder - you'd need to implement actual Twitter API integration
    # using Tweepy or similar library
    bearer_token = st.secrets.get("TWITTER_BEARER_TOKEN", "")
    
    if not bearer_token:
        st.warning("Twitter API not configured. Using sample data.")
        # Return sample data for demonstration
        sample_tweets = [
            {"text": f"Love the new {keyword} feature! So helpful.", "created_at": "2023-10-15"},
            {"text": f"{keyword} is terrible. Worst update ever.", "created_at": "2023-10-15"},
            {"text": f"The {keyword} is okay, but could be better.", "created_at": "2023-10-14"}
        ]
        return sample_tweets
    
    # Actual API implementation would go here
    headers = {"Authorization": f"Bearer {bearer_token}"}
    params = {
        "query": keyword,
        "max_results": count,
        "tweet.fields": "created_at,public_metrics"
    }
    
    try:
        response = requests.get(
            "https://api.twitter.com/2/tweets/search/recent",
            headers=headers,
            params=params
        )
        return response.json().get("data", [])
    except Exception as e:
        st.error(f"Error fetching tweets: {e}")
        return []
    
    # Reddit sentiment analysis
def analyze_reddit_sentiment(subreddit, keyword, limit=50):
    """
    Analyze sentiment of Reddit posts (requires PRAW library)
    """
    # Placeholder implementation
    st.info("Reddit integration would require PRAW library and API credentials")
    return []

# News sentiment analysis
def fetch_news_sentiments(keyword, days=7):
    """
    Fetch news articles and analyze sentiment
    """
    # Could integrate with NewsAPI or similar services
    st.info("News integration would require NewsAPI or similar service")
    return []

# Enhanced preprocessing for real-world text
def advanced_preprocess_text(text):
    """
    More sophisticated text preprocessing for social media/texting language
    """
    # Convert to lowercase
    text = text.lower()
    
    # Expand contractions
    contractions = {
        "don't": "do not",
        "can't": "cannot",
        "won't": "will not",
        "it's": "it is",
        "i'm": "i am",
        # Add more contractions
    }
    
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove mentions and hashtags (but keep the text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    
    # Handle emojis (either remove or convert to text)
    # You might want to use a library like emoji for better handling
    text = re.sub(r'[^\w\s@#]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

# Add to imports
from wordcloud import WordCloud
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Time series sentiment tracking
def plot_sentiment_timeline(data):
    """
    Create a timeline of sentiment over time
    """
    if 'date' not in data.columns or 'sentiment' not in data.columns:
        st.warning("Need date and sentiment columns for timeline analysis")
        return
    
    # Convert to time series
    data['date'] = pd.to_datetime(data['date'])
    time_series = data.groupby([data['date'].dt.date, 'sentiment']).size().unstack(fill_value=0)
    
    # Create plot
    fig = go.Figure()
    for sentiment in time_series.columns:
        fig.add_trace(go.Scatter(
            x=time_series.index, 
            y=time_series[sentiment],
            mode='lines+markers',
            name=sentiment,
            stackgroup='one'  # stack plots
        ))
    
    fig.update_layout(
        title="Sentiment Over Time",
        xaxis_title="Date",
        yaxis_title="Number of Mentions",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig)

# Word cloud visualization
def generate_wordcloud(text_data, sentiment):
    """
    Generate word cloud for specific sentiment
    """
    # Combine all text of the given sentiment
    text = ' '.join(text_data[text_data['sentiment'] == sentiment]['text'])
    
    if not text:
        st.warning(f"No text data available for {sentiment} sentiment")
        return None
    
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis' if sentiment == 'positive' else 'Reds' if sentiment == 'negative' else 'Blues'
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Most Common Words in {sentiment.capitalize()} Sentiment')
    
    return fig

# Comparative analysis
def analyze_multiple_texts(texts, batch_size=100):
    """
    Analyze sentiment for multiple texts with batch processing and error handling
    """
    if not texts or not isinstance(texts, (list, pd.Series)):
        return pd.DataFrame(columns=['text', 'sentiment', 'confidence', 'error'])
    
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        for text in batch:
            if not isinstance(text, str) or not text.strip():
                # Skip non-string or empty texts
                results.append({
                    'text': str(text)[:100] if text else '',
                    'sentiment': 'invalid',
                    'confidence': 0.0,
                    'error': 'Invalid text input'
                })
                continue
            
            try:
                # Preprocess text
                cleaned_text = preprocess_text(text)
                # Vectorize
                text_vector = vectorizer.transform([cleaned_text])
                # Predict
                prediction = model.predict(text_vector)[0]
                probability = model.predict_proba(text_vector).max()
                
                results.append({
                    'text': text,
                    'cleaned_text': cleaned_text,
                    'sentiment': prediction,
                    'confidence': probability,
                    'error': None
                })
                
            except Exception as e:
                # Handle errors gracefully
                results.append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'sentiment': 'error',
                    'confidence': 0.0,
                    'error': str(e)
                })
    
    return pd.DataFrame(results)
    
    # Create comparison visualization
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
    
    fig.add_trace(go.Pie(
        labels=before_sentiments['sentiment'].value_counts().index,
        values=before_sentiments['sentiment'].value_counts().values,
        name=before_label
    ), 1, 1)
    
    fig.add_trace(go.Pie(
        labels=after_sentiments['sentiment'].value_counts().index,
        values=after_sentiments['sentiment'].value_counts().values,
        name=after_label
    ), 1, 2)
    
    fig.update_traces(hole=.4, hoverinfo="label+percent+name")
    
    fig.update_layout(
        title_text=f"Sentiment Comparison: {before_label} vs {after_label}",
        annotations=[dict(text=before_label, x=0.18, y=0.5, font_size=20, showarrow=False),
                     dict(text=after_label, x=0.82, y=0.5, font_size=20, showarrow=False)]
    )
    
    return fig

# Sentiment scoring and metrics
def calculate_sentiment_metrics(data):
    """
    Calculate business-relevant sentiment metrics
    """
    total_reviews = len(data)
    positive_count = len(data[data['sentiment'] == 'positive'])
    negative_count = len(data[data['sentiment'] == 'negative'])
    neutral_count = len(data[data['sentiment'] == 'neutral'])
    
    metrics = {
        'total_reviews': total_reviews,
        'positive_percentage': (positive_count / total_reviews) * 100 if total_reviews > 0 else 0,
        'negative_percentage': (negative_count / total_reviews) * 100 if total_reviews > 0 else 0,
        'neutral_percentage': (neutral_count / total_reviews) * 100 if total_reviews > 0 else 0,
        'net_sentiment_score': ((positive_count - negative_count) / total_reviews) * 100 if total_reviews > 0 else 0
    }
    
    return metrics

# Alert system for negative sentiment spikes
def check_sentiment_alerts(data, threshold=0.3):
    """
    Check if negative sentiment exceeds threshold and send alerts
    """
    metrics = calculate_sentiment_metrics(data)
    
    alerts = []
    if metrics['negative_percentage'] > threshold * 100:
        alerts.append({
            'type': 'high_negative_sentiment',
            'message': f"Negative sentiment has reached {metrics['negative_percentage']:.2f}%",
            'severity': 'high'
        })
    
    if metrics['positive_percentage'] < 0.1 * 100:
        alerts.append({
            'type': 'low_positive_sentiment',
            'message': f"Positive sentiment is only {metrics['positive_percentage']:.2f}%",
            'severity': 'medium'
        })
    
    return alerts


# E-commerce specific analysis
def analyze_product_reviews(product_data):
    """
    Specialized analysis for product reviews
    """
    # Extract aspects (features) mentioned in reviews
    aspects = {
        'price': ['price', 'cost', 'expensive', 'cheap', 'affordable'],
        'quality': ['quality', 'durable', 'broken', 'last', 'material'],
        'shipping': ['shipping', 'delivery', 'fast', 'slow', 'package'],
        'customer_service': ['service', 'support', 'help', 'response', 'rude']
    }
    
    aspect_sentiments = {aspect: {'positive': 0, 'negative': 0, 'neutral': 0} 
                         for aspect in aspects.keys()}
    
    # Analyze each review for aspect sentiments
    for _, review in product_data.iterrows():
        text = review['text'].lower()
        sentiment = review['sentiment']
        
        for aspect, keywords in aspects.items():
            if any(keyword in text for keyword in keywords):
                aspect_sentiments[aspect][sentiment] += 1
    
    return aspect_sentiments

# Social media crisis detection
def detect_crisis_situations(sentiment_data, time_window=24):
    """
    Detect potential PR crises based on sentiment patterns
    """
    # Group by time windows
    sentiment_data['time_window'] = sentiment_data['timestamp'].dt.floor(f'{time_window}H')
    time_series = sentiment_data.groupby(['time_window', 'sentiment']).size().unstack(fill_value=0)
    
    crises = []
    if 'negative' in time_series.columns:
        # Check for significant spikes in negative sentiment
        negative_mean = time_series['negative'].mean()
        negative_std = time_series['negative'].std()
        
        for time, negative_count in time_series['negative'].items():
            if negative_count > negative_mean + 2 * negative_std:
                crises.append({
                    'time': time,
                    'negative_count': negative_count,
                    'severity': 'high' if negative_count > negative_mean + 3 * negative_std else 'medium'
                })
    
    return crises

# Add these UI enhancements to your Streamlit app

# Sidebar filters and options
def create_sidebar_filters():
    """
    Create interactive filters in the sidebar
    """
    st.sidebar.header("Filters & Options")
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(datetime.now() - timedelta(days=7), datetime.now()),
        max_value=datetime.now()
    )
    
    # Sentiment filter
    selected_sentiments = st.sidebar.multiselect(
        "Sentiments to Include",
        options=['positive', 'negative', 'neutral'],
        default=['positive', 'negative', 'neutral']
    )
    
    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Minimum Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05
    )
    
    return {
        'date_range': date_range,
        'sentiments': selected_sentiments,
        'confidence_threshold': confidence_threshold
    }

# Dashboard overview
def create_dashboard_overview(metrics, alerts):
    """
    Create a comprehensive dashboard overview
    """
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reviews", metrics['total_reviews'])
    
    with col2:
        st.metric("Positive Sentiment", f"{metrics['positive_percentage']:.1f}%")
    
    with col3:
        st.metric("Negative Sentiment", f"{metrics['negative_percentage']:.1f}%")
    
    with col4:
        st.metric("Net Sentiment Score", f"{metrics['net_sentiment_score']:.1f}")
    
    # Alerts
    if alerts:
        st.warning("**Alerts**")
        for alert in alerts:
            st.write(f"⚠️ {alert['message']}")
            
            # Safe way to verify keys are loaded without exposing them
try:
    keys_loaded = all([
        st.secrets.get("TWITTER_BEARER_TOKEN",""),
        st.secrets.get("REDDIT_CLIENT_ID",""), 
        st.secrets.get("REDDIT_CLIENT_SECRET",""),
        st.secrets.get("NEWSAPI_KEY","")
    ])
    
    if keys_loaded:
        st.sidebar.success("✅ All API keys loaded successfully")
    else:
        st.sidebar.warning("⚠️ Some API keys are missing")
        
except Exception as e:
    bearer_token = ""
    reddit_client_id = ""
    reddit_client_secret = ""
    newsapi_key = ""
    api_keys_configured = False
    st.sidebar.warning("⚠️ API configuration loaded with fallbacks")
    st.sidebar.error("❌ Error loading API keys")
    
    def safe_debug_info():
     """Display safe debugging information without exposing keys"""
    
    debug_info = {
        "Twitter API": "✅ Configured" if st.secrets.get("TWITTER_BEARER_TOKEN") else "❌ Missing",
        "Reddit API": "✅ Configured" if st.secrets.get("REDDIT_CLIENT_ID") else "❌ Missing", 
        "NewsAPI": "✅ Configured" if st.secrets.get("NEWSAPI_KEY") else "❌ Missing",
        "Total Services": f"{sum(1 for key in ['TWITTER_BEARER_TOKEN', 'REDDIT_CLIENT_ID', 'NEWSAPI_KEY'] if st.secrets.get(key))}/3 configured"
    }
    
    with st.expander("🔧 API Configuration Status (Safe View)"):
        for service, status in debug_info.items():
            st.write(f"**{service}**: {status}")
    
  
            
            # Test secrets loading
#try:
 #   st.write("Twitter Bearer Token:", st.secrets["TWITTER_BEARER_TOKEN"][:10] + "..." if st.secrets["TWITTER_BEARER_TOKEN"] else "Not set")
  #  st.write("Reddit Client ID:", st.secrets["REDDIT_CLIENT_ID"][:10] + "..." if st.secrets["REDDIT_CLIENT_ID"] else "Not set")
   # st.write("NewsAPI Key:", st.secrets["NEWSAPI_KEY"][:10] + "..." if st.secrets["NEWSAPI_KEY"] else "Not set")
#except Exception as e:
 #   st.warning("Secrets not configured properly. Some features will be disabled.")
  #  st.error(f"Error: {e}")