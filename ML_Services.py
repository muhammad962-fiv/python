from fastapi import FastAPI
from pydantic import BaseModel

# --- Keyword Extraction ---
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

# --- Sentiment ---
from transformers import pipeline

# --- LLM (Gemini) ---
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # or "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Keyword extraction models
embedder = SentenceTransformer("all-MiniLM-L6-v2")
keyword_model = KeyBERT(model=embedder)

# Sentiment models
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)


from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-3.1-flash-lite")

# ------------------------------
# ----- SCHEMA CLASSES ---------
# ------------------------------

class KeywordRequest(BaseModel):
    text: str
    num_keywords: int = 1000

class SentimentRequest(BaseModel):
    text: str

class ReportRequest(BaseModel):
    data: dict

# ------------------------------
# ---- ENDPOINTS ---------------
# ------------------------------


@app.post("/extract_keywords")
def extract_keywords(req: KeywordRequest):
    text = req.text
    

    max_chars = 2000
    chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

    all_keywords = []

    for chunk in chunks:
        if len(chunk.strip()) < 60:
            continue

        try:
            keywords = keyword_model.extract_keywords(
                chunk,
                keyphrase_ngram_range=(1, 2),
                stop_words="english",
                top_n=10
            )

            all_keywords.extend([kw for kw, _ in keywords])

        except Exception:
            continue

    # nothing extracted at all
    if len(all_keywords) == 0:
        return {
            "keywords": ["shit_no_keywords"]
        }


    return {
        "keywords": all_keywords
    }

@app.post("/analyze_sentiment")
def analyze_sentiment(req: SentimentRequest):
    text = req.text

    # split into safe chunks (model limit safe zone)
    max_chars = 2000
    chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

    scores = []

    for chunk in chunks:
        result = sentiment_model(chunk[:512])[0]  # extra safety
        score = result["score"]

        # convert to positive scale [0,1]
        if result["label"] == "NEGATIVE":
            score = 1 - score

        scores.append(score)

    # aggregate global sentiment
    window = scores[-10:]
    if len(window) == 0:
        final_score = 0.0
    else:
        final_score = sum(window) / len(window)

    return {
        "score": final_score,
        "label": "AGGREGATED"
    }
    
@app.post("/generate_strategy_report")
def generate_strategy_report(req: ReportRequest):
    prompt = f"""
Analyze the following SEO and competitor data for a website.

Tasks:
1. Compare authority scores of brand and competitors
2. Identify strongest competitors
3. Estimate backlinks needed to compete
4. Suggest SEO strategy (3–4 bullet points, specific)
5. Suggest content strategy (3–4 bullet points, specific)
6. Suggest backlink building strategy (3–4 bullet points)
7. Suggest social media plan
8. Suggest Google Ads campaigns

and must Return a markdown text for the whole report, proper 

DATA (JSON):
{req.data}
"""
    response = gemini_model.generate_content(prompt)
    return {"strategy_report": response.text}
