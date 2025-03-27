import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load pretrained model and tokenizer
MODEL_NAME = "cardiffnlp/twitter-roberta-base-offensive"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

def predict_comment(text):
    """Predict whether a comment is hate speech, offensive, or normal."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    label_map = {0: "Normal", 1: "Offensive", 2: "Hate Speech"}  # Updated for three categories
    return label_map.get(prediction, "Unknown")