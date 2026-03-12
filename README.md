# Journal of Chemical Information and Modeling Text Generation & Fine-Tuning Toolkit

This repository provides a complete pipeline for collecting, preparing, fine-tuning, and serving a GPT-2-based language model on open-access literature from the Journal of Chemical Information and Modeling, specifically from Europe PMC. It includes scripts for data fetching, preprocessing, model training, and a web application (frontend and backend) for interacting with your fine-tuned model.

## Features

- **Data Collection**: Fetches open-access articles from the Journal of Chemical Information and Modeling via Europe PMC.
- **Data Preparation**: Cleans and tokenizes text, saving as JSONL for efficient model training.
- **Model Fine-Tuning**: Fine-tunes GPT-2 (or compatible models) on the prepared Journal of Chemical Information and Modeling dataset using HuggingFace Transformers.
- **Web Application Backend**: FastAPI backend for serving the fine-tuned model and providing a `/predict` endpoint for text generation.
- **Web Application Frontend**: React + TypeScript + Vite frontend for user interaction with the model.
- **CLI Demo**: Simple command-line interface to interact with the API.

## Directory Structure

- `scripts/`: Utility scripts for data fetching, preparation, and model training.
- `webapp/`: Front-end and back-end for interfacing with trained models.
- `requirements.txt`: Python dependencies.
- `run_workflow.ipynb`: Run the end-to-end workflow (fetch/prepare data, model training)

## Usage

### 1. Fetch Data

```bash
python scripts/fetch.py --page_size 100 --output_dir data/europepmc/
```

### 2. Prepare Data

```bash
python scripts/prepare_text.py --data_dir data/europepmc/ --output_jsonl data/europepmc_prepared.jsonl
```

### 3. Fine-Tune Model

```bash
python scripts/finetune.py --data_path data/europepmc_prepared.jsonl --output_dir models/gpt2-finetuned-europepmc/
```

## Web Application

The `webapp/` directory provides a modern web interface for interacting with your fine-tuned model. It consists of a **backend** (FastAPI) for model inference and a **frontend** (React + TypeScript + Vite) for user interaction.

### Backend (FastAPI)

- Located in `webapp/backend/`.
- Loads the fine-tuned model from `models/gpt2-finetuned-europepmc/` (adjustable in `main.py`).
- Provides a `/predict` endpoint for text generation using the model.
- Handles CORS to allow requests from the frontend.

**Usage:**

```bash
cd webapp/backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at `http://localhost:8000`.

### Frontend (React + TypeScript + Vite)

- Located in `webapp/frontend/`.
- Provides a simple UI for entering prompts and viewing model responses.
- Communicates with the backend at `http://localhost:8000/predict`.

**Usage:**

```bash
cd webapp/frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:3000`.

## Requirements

See `requirements.txt` for all dependencies. Key packages:
- `transformers`
- `torch`
- `fastapi`
- `uvicorn`
- `requests`
- `scikit-learn`
- `datasets`
- `tqdm`
- `bs4`
- `lxml`
