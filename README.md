Question Answering API – FastAPI Deployment
This project wraps a fine-tuned Question Answering model into a FastAPI application for easy interaction and inference.
Steps:
1.Download the Model
Run the following command to download and cache the model locally:

'python download_model.py'

This will download the required model and tokenizer to your local system.

2.Start the API Server
Use uvicorn to start the FastAPI server:

'uvicorn app:app --reload'

3.Access the Swagger UI
Once the server is running, open your browser and navigate to:
'http://127.0.0.1:8000/docs'

This interactive UI allows you to test the /predict endpoint directly.

