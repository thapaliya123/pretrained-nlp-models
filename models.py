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
    - pip install torch
    - pip install transformers
"""
import pandas as pd
from transformers import pipeline

class Pretrained_NLP:
    def __init__(self, text) -> None:
        self.text = text
        pass

    def sentiment_analysis(self) -> None:
        """
        compute sentiments (positive or negative) of text using HuggingFace pipeline api.
        By default it will use DistilBERT pretrained model.

        Arguments:
        -------------
        self: class object
        
        Returns:
        ------------------
            df_text_sentiment: sentiment of text, pd.DataFrame
                
        """
        # pass 'text-classification' or 'sentitment-analysis'
        # for sentiment analysis
        nlp_task = 'text-classification' # or nlp_task = 'sentiment-analysis'

        # instantiate pipeline api,
        # pass nlp_task as an arugment,
        # it will downloads pretrained models
        # from huggingface ecosystem, and cached it
        classifier = pipeline(nlp_task)

        # pass text, for sentiment analysis
        text_sentiment = classifier(self.text)

        # convert to pandas dataframe
        # for readable output
        df_text_sentiment = pd.DataFrame(text_sentiment)

        return df_text_sentiment
        
    def named_entity_recognition(self) -> None:
        classifier = pipeline('ner')
        classifier(self.text)
    def question_answering(self) -> None:
        pass
    def summarization(self) -> None:
        pass
    def translation(self) -> None:
        pass
    def text_generation(self) -> None:
        pass

def main():
    text = """Dear Amazon, last week I ordered an Optimus Prime action figure \
              from your online store in Germany. Unfortunately, when I opened the package, \
              I discovered to my horror that I had been sent an action figure of Megatron \
              instead! As a lifelong enemy of the Decepticons, I hope you can understand my \
              dilemma. To resolve the issue, I demand an exchange of Megatron for the \
              Optimus Prime figure I ordered. Enclosed are copies of my records concerning \
              this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""

    
    pretrained_nlp = Pretrained_NLP(text)
    df_text_sentiment = pretrained_nlp.sentiment_analysis()
    print(df_text_sentiment)

if __name__ == "__main__":
    main() 

