# Model Inference Web App

## Backend (FastAPI)

1. Create and activate a Python environment (e.g., conda or venv).
2. Install dependencies:
   cd webapp/backend
   pip install -r requirements.txt
3. Run the API server:
   uvicorn main:app --reload --host 0.0.0.0 --port 8000

## Frontend (React + TypeScript + Vite)

1. Install dependencies:
   cd webapp/frontend
   npm install
2. Start the development server:
   npm run dev

The frontend will be available at http://localhost:3000 and will communicate with the backend at http://localhost:8000.

---

- The backend loads the fine-tuned model from models/gpt2-finetuned-europepmc/.
- You can adjust the model path or inference logic in backend/main.py as needed.
- The frontend provides a simple UI for entering prompts and viewing model responses.
