import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load model and tokenizer
MODEL_PATH = "app/hate_offense_bert_model"  # Update with actual path

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

def predict_comment(text):
    """Predict whether a comment is hate speech, offensive, or normal."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    label_map = {0: "Hate Speech", 1: "Offensive", 2: "Normal"}
    return label_map[prediction]
