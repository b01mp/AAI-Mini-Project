import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from transformers import pipeline

# Download necessary NLTK data files for text processing
nltk.download('punkt')
nltk.download('stopwords')

class TextSummarizer:
    def __init__(self):
        """
        Initialize the summarizer. 
        We use the T5 model, which is a Transformer model (Syllabus Topic 2).
        Using 't5-small' ensures it runs smoothly on standard student laptops.
        """
        print("Loading Transformer Model (T5-Small)... This may take a moment on the first run.")
        self.abstractive_summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")
        print("Model loaded successfully!\n")

    def extractive_summary(self, text, num_sentences=3):
        """
        Extractive Summarization using TF-IDF Feature Extraction (Syllabus Topic 1).
        Selects the most important existing sentences based on word frequencies.
        """
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text

        # Feature Extraction using TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Calculate sentence scores based on the sum of their TF-IDF values
        sentence_scores = np.asarray(tfidf_matrix.sum(axis=1)).flatten()
        
        # Get the indices of the top N scoring sentences
        top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
        
        # Sort indices to maintain the original chronological order of the text
        top_indices = sorted(top_indices)
        
        # Join the selected sentences
        summary = ' '.join([sentences[i] for i in top_indices])
        return summary

    def abstractive_summary(self, text, min_length=30, max_length=130):
        """
        Abstractive Summarization using Neural Transformer Model (Syllabus Topic 2).
        Generates new sentences that capture the core meaning.
        """
        # T5 requires a specific task prefix to know what to do
        input_text = "summarize: " + text
        
        summary = self.abstractive_summarizer(
            input_text, 
            min_length=min_length, 
            max_length=max_length, 
            do_sample=False
        )
        return summary[0]['summary_text']


if __name__ == "__main__":
    # --- Test the application ---
    sample_text = """
    Artificial Intelligence (AI) is transforming various sectors, including business and healthcare. 
    In business, AI is used for smart investment analysis, customer support, and robotic process automation. 
    Natural Language Processing (NLP), a subset of AI, enables machines to understand human language. 
    Techniques like text classification, sentiment analysis, and named entity recognition are fundamental to NLP. 
    Furthermore, advanced neural models such as RNNs, LSTMs, and Transformers like BERT and GPT have revolutionized 
    how machines process and generate text. These models are now being applied to create sophisticated text summarizers, 
    capable of condensing large documents into concise, meaningful summaries.
    """
    
    # Initialize the app
    app = TextSummarizer()
    
    print("--- Extractive Summary (Topic 1: Feature Extraction) ---")
    print(app.extractive_summary(sample_text, num_sentences=2))
    
    print("\n--- Abstractive Summary (Topic 2: Transformer Models) ---")
    print(app.abstractive_summary(sample_text))