from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Use quantization-aware loading if needed
def load_quantized_model(path: str):
    model = AutoModelForQuestionAnswering.from_pretrained(path, torch_dtype=torch.qint8)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return pipeline("question-answering", model=model, tokenizer=tokenizer)

# Fallback to standard
def load_model(path: str):
    try:
        return load_quantized_model(path)
    except Exception:
        print("Falling back to non-quantized model")
        return pipeline("question-answering", model=path, tokenizer=path)

model_path = r"C:\D drive\qa_app\qa_app\model"
qa_pipeline = load_model(model_path)

def get_answer(context: str, question: str):
    return qa_pipeline({
        "context": context,
        "question": question
    })
