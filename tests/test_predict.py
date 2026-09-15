import pytest
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

class SentimentAnalyzer:
    def __init__(self, model_name: str = MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.labels = ["negative", "neutral", "positive"]

    def preprocess_text(self, text: str) -> str:
        text = re.sub(r"http\S+|www\S+|https\S+", "http", text, flags=re.MULTILINE)
        text = re.sub(r"@\w+", "@user", text)
        return text.strip()

    def predict(self, text: str) -> dict:
        cleaned_text = self.preprocess_text(text)
        inputs = self.tokenizer(cleaned_text, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = outputs.logits[0]
            probabilities = torch.softmax(scores, dim=0)

        confidence, predicted_class_id = torch.max(probabilities, dim=0)
        predicted_label = self.labels[predicted_class_id.item()]

        return {
            "text": text,
            "cleaned_text": cleaned_text,
            "sentiment": predicted_label,
            "confidence": round(confidence.item(), 4),
            "probabilities": {
                label: round(prob.item(), 4) for label, prob in zip(self.labels, probabilities)
            }
        }

@pytest.fixture(scope="module")
def analyzer():
    return SentimentAnalyzer()

def test_preprocessing(analyzer):
    raw_text = "Check this link https://example.com and mention @john!"
    cleaned = analyzer.preprocess_text(raw_text)
    assert "https://example.com" not in cleaned
    assert "@john" not in cleaned
    assert "http" in cleaned
    assert "@user" in cleaned

def test_predict_output_structure(analyzer):
    result = analyzer.predict("This product is amazing!")
    assert isinstance(result, dict)
    assert "sentiment" in result
    assert "confidence" in result

def test_sentiment_values(analyzer):
    result = analyzer.predict("Great service!")
    assert result["sentiment"] in ["negative", "neutral", "positive"]
    assert 0.0 <= result["confidence"] <= 1.0
