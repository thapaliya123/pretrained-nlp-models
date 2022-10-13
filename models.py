"""
- Solving various NLP task using transformers library from HuggingFace pretrained models.
- NLP task includes:
    - Sentiment Analysis
    - Named Entity Recognition
    - Question and Answering
    - Summarization
    - Translation
    - Text Generation
- Requirements
    - pip install pandas
    - pip install transformers
"""
import pandas as pd
from transformers import pipeline

class Pretrained_NLP:
    def __init__(self, text) -> None:
        self.text = text
        pass
    def sentiment_analysis(self) -> None:
        classifier = pipeline('text-classification')
        text_sentiment = classifier(self.text)
        df_text_sentiment = pd.DataFrame(text_sentiment)
        
    def named_entity_recognition(self) -> None:
        pass
    def question_answering(self) -> None:
        pass
    def summarization(self) -> None:
        pass
    def translation(self) -> None:
        pass
    def text_generation(self) -> None:
        pass

def main():
    pass

if __name__ == "__main__":
    main() 

