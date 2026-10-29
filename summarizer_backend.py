import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Download necessary NLTK data files for text processing (quiet=True prevents spammy output)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

class TextSummarizer:
    def __init__(self):
        """
        Initialize the summarizer. 
        We use the T5 model, a Transformer neural network (Syllabus Topic 2).
        Direct model loading bypasses recent pipeline deprecations in HuggingFace.
        """
        print("Loading Transformer Model (T5-Small)... This may take a moment on the first run.")
        model_name = "t5-small"
        
        # Load tokenizer and model directly
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Automatically use GPU if available, otherwise fallback to CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Model loaded successfully on {self.device.upper()}!\n")

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
        
        # 1. Tokenize the input text and move to the correct device (CPU/GPU)
        inputs = self.tokenizer.encode(
            input_text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).to(self.device)
        
        # 2. Generate summary using Beam Search (produces higher quality than basic sampling)
        summary_ids = self.model.generate(
            inputs,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        
        # 3. Decode the generated token IDs back into human-readable text
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary