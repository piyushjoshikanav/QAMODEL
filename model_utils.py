from transformers import pipeline

# Use raw string (r"...") to avoid escape issues in Windows paths
model_path = r"C:\D drive\qa_app\qa_app\model"

qa_pipeline = pipeline("question-answering", model=model_path, tokenizer=model_path)

def get_answer(context: str, question: str):
    return qa_pipeline({
        "context": context,
        "question": question
    })
