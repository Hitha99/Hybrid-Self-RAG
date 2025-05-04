# Hybrid Self-RAG

A novel framework combining **Retrieval-Augmented Generation (RAG)**, **self-critique**, and **external knowledge validation** to improve the factual reliability of large language models (LLMs) and significantly reduce hallucinations.

---

## 📖 Overview

Large Language Models (LLMs) are powerful but prone to generating hallucinated (factually incorrect) outputs, especially in critical domains like healthcare, law, and scientific research.  
This project introduces a **Hybrid Self-RAG** system that:

- Enhances self-reflection with **structured external validation** (Wikipedia, ArXiv APIs)
- Calculates **FactScores** based on semantic similarity to retrieved evidence
- Triggers **fallback dense retrieval** using FAISS when external sources are insufficient
- Reduces hallucination rates while maintaining generation fluency

---

## 🏗️ Architecture

- **Initial Generation:** Language model (e.g., GPT-4) generates a first response
- **Self-Reflection:** Self-critiques and selects the most consistent version
- **External Validation:** Factual claims are extracted and checked live against Wikipedia and ArXiv
- **Fact Scoring:** Semantic similarity scores determine factual reliability
- **Fallback Retrieval:** If validation confidence is low, dense retrieval fetches relevant context from a local FAISS index
- **Regeneration:** Low-confidence outputs are corrected and regenerated with evidence

---

## 📈 Results

| Metric | Baseline (Self-RAG Only) | Hybrid Self-RAG |
|:---|:---|:---|
| **Factual Accuracy** | 81% | 89% |
| **Hallucination Rate** | 19% | 11% |
| **Average FactScore** | 0.65 | 0.87 |

✅ Demonstrated significant improvements in factual correctness, with up to **34% better semantic alignment**.

---

## 🛠️ Tech Stack

- **Language Models:** GPT-4 (via API)
- **Embedding Models:** SentenceTransformers (all-MiniLM-L6-v2)
- **Retrieval:** Wikipedia API, ArXiv API, FAISS (fallback dense retriever)
- **Libraries:** HuggingFace Transformers, Sentence-Transformers, FAISS, spaCy, vLLM
- **Hardware:** NVIDIA A100 GPU (Texas A&M HPRC)

---

## 📚 How to Run

```bash
# Clone the repository
git clone git@github.com:Hitha99/Hybrid-Self-RAG.git

# Install dependencies
pip install -r requirements.txt

# Configure your API keys (Wikipedia/ArXiv if needed)

# Run the pipeline
python run_hybrid_selfrag.py
