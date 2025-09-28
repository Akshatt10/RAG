# FastAPI Application

This is a FastAPI application. The main file is located in the `notebook` folder.

## Requirements

Make sure you have Python 3.10+ installed and `pip` available. Install dependencies:

```bash
pip install -r requirements.txt
```

# Running the Application

```bash
cd notebook
```

## Then run:
```bash
uvicorn main:app --reload
```
main refers to main.py.

app is the FastAPI instance inside main.py.

--reload enables automatic reloading on code changes (useful for development).

## Open your browser and go to:
```bash
http://127.0.0.1:8000
```
