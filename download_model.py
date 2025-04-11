from transformers import pipeline

from huggingface_hub import login

# ⏳ Login first
login("")
# 🧠 Replace with your fine-tuned model name if needed
model_name = "piyushjoshi/qa-custom-model"
# model_name = "your-username/your-fine-tuned-model"  # <- if you have a custom one

# Load the model and tokenizer
print(f"⏳ Downloading model '{model_name}' from Hugging Face...")
qa_pipeline = pipeline("question-answering", model=model_name, tokenizer=model_name)

# Save to local directory
save_path = "./qa_app/model"
print(f"📁 Saving model to {save_path}")
qa_pipeline.model.save_pretrained(save_path)
qa_pipeline.tokenizer.save_pretrained(save_path)

print("✅ Model downloaded and saved successfully!")
