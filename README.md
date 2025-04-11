# 📘 Question Answering API – FastAPI Deployment

This project deploys a fine-tuned Hugging Face Question Answering model using **FastAPI**. It provides an interactive REST API for inference and a **Streamlit-based UI** for a user-friendly frontend. Performance monitoring is integrated using **Weights & Biases (W&B)**, and the codebase is optimized with asynchronous processing and robust error handling.

---

##  Features

-  Hugging Face Transformers model for QA
-  Async FastAPI endpoint with response latency tracking
-  W&B integration for logging score and inference latency
-  Streamlit frontend for non-technical users
-  Pydantic-based request validation
-  Swagger UI for live API testing
-  Modular code structure for easy scaling

---

##  Project Architecture

[Streamlit UI] --> [FastAPI Endpoint (/predict)] --> [HF QA Pipeline] --> [Answer + Score]
                                      |
                               [W&B Performance Logs]

**Tech Stack:**
        Python 3.11
        FastAPI
        Hugging Face Transformers
        Weights & Biases (wandb)
        Streamlit
        Pydantic

**Setup Instructions**

1.Clone the Repository  

   ```bash
    git clone <repo_url>
    cd qa_app
   ```

2.Create Virtual Environment

    ```bash
    python -m venv qa
    source qa/bin/activate  # On Windows: qa\Scripts\activate

    ```
3.Install Dependencies

    ```bash
    pip install -r requirements.txt

    ```
4.Download Model Locally
This will download the model and tokenizer into a model/ folder:

    ```bash
    python download_model.py
    ```

5.Start FastAPI Server
```bash
    uvicorn app:app --reload

```

6.Run the UI:
```bash
    streamlit run app_ui.py
```

**UI Preview:**
    Enter context and question in the provided boxes
    Click submit to view the generated answer
    See response and score in real-time

   #  Monitoring with Weights & Biases
        The app logs:
            -Inference Latency (ms)
            -Model Confidence Score
        You can explore trends across multiple requests for benchmarking and analysis.
        wandb dashboard link will be generated on first run

**Folder Structure:**    
qa_app/
    │
    ├── app.py               # FastAPI app with async logic
    ├── app_ui.py            # Streamlit UI
    ├── model_utils.py       # Loads Hugging Face pipeline
    ├── download_model.py    # Script to fetch model/tokenizer
    ├── requirements.txt
    ├──FineTuningQAModel.ipynb # Jupyter notebook for model fine-tuning
    ├── README.md
    └── model/               # Locally saved model folder
    