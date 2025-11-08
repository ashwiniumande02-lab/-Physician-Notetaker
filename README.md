# 🩺 Physician Notetaker AI Pipeline  
**Author:** Ashwini Umande  
**Email:** ashwiniumande02@gmail.com  

---

## 📘 Project Overview
This project builds an **AI-powered NLP pipeline** for medical transcription analysis — designed to convert doctor–patient conversations into structured, clinically useful summaries.  
It includes **medical entity extraction**, **sentiment analysis**, **intent detection**, and **SOAP note generation**.

---

## 🧠 Objectives
- Extract medical details (symptoms, diagnosis, treatment, prognosis) from transcribed text  
- Analyze **patient sentiment** and **intent**  
- Generate structured clinical documentation (SOAP notes)

---

## ⚙️ Technologies Used
| Component | Tool / Library | Purpose |
|------------|----------------|----------|
| **NLP Engine** | spaCy | Tokenization, Entity Extraction |
| **Transformer Models** | BERT, DistilBERT, BART | Summarization & Sentiment |
| **Text Processing** | NLTK, TextBlob | Preprocessing & Sentiment |
| **Machine Learning** | scikit-learn | Classification & Modeling |
| **Platform** | Google Colab / Jupyter Notebook | Development & Testing |

---

## 🧩 Pipeline Components

### **1. Medical NLP Summarization**
Extracts and summarizes medical data from transcripts.

**Key Deliverables:**
- Named Entity Recognition (NER)
- Text Summarization
- Keyword Extraction

Example Output**
```json
{
  "Patient_Name": "Janet Jones",
  "Symptoms": ["Neck pain", "Back pain", "Head impact"],
  "Diagnosis": "Whiplash injury",
  "Treatment": ["10 physiotherapy sessions", "Painkillers"],
  "Current_Status": "Occasional backache",
  "Prognosis": "Full recovery expected within six months"
}

 2. Sentiment & Intent Analysis

Uses Transformer-based models (like BERT/DistilBERT) to detect:

Sentiment: Anxious, Neutral, or Reassured

Intent: Seeking reassurance, Reporting symptoms, Expressing concern

Example Output


{
  "Sentiment": "Anxious",
  "Intent": "Seeking reassurance"
}


3. SOAP Note Generation (Bonus)
Automatically generates a structured SOAP note (Subjective, Objective, Assessment, Plan).
Example Output
{
  "Subjective": {
    "Chief_Complaint": "Neck and back pain",
    "History_of_Present_Illness": "Patient had a car accident, experienced pain for four weeks, now occasional back pain."
  },
  "Objective": {
    "Physical_Exam": "Full range of motion, no tenderness.",
    "Observations": "Patient appears in good health."
  },
  "Assessment": {
    "Diagnosis": "Whiplash injury and back strain",
    "Severity": "Mild, improving"
  },
  "Plan": {
    "Treatment": "Continue physiotherapy as needed, use painkillers for relief.",
    "Follow-Up": "Return if pain worsens or persists beyond six months."
  }
}


🧩 Example Code: Entity Extraction
import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_sm")

def extract_medical_entities(text):
    """
    Extracts Symptoms, Diagnosis, Treatment, and Prognosis
    from patient transcripts using spaCy and rule-based matching.
    """
    doc = nlp(text)

    categories = {
        "Symptoms": [],
        "Diagnosis": [],
        "Treatment": [],
        "Prognosis": []
    }

    patterns = {
        "Symptoms": ["neck pain", "back pain", "headache", "stiffness", "discomfort", "ache", "pain"],
        "Diagnosis": ["whiplash", "fracture", "sprain", "strain", "concussion", "whiplash injury"],
        "Treatment": ["physiotherapy", "painkillers", "therapy", "session", "medication"],
        "Prognosis": ["recovery", "improvement", "healing", "no long-term damage"]
    }

    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for label, terms in patterns.items():
        matcher.add(label, [nlp.make_doc(term) for term in terms])

    matches = matcher(doc)
    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id]
        span = doc[start:end].text
        if span not in categories[label]:
            categories[label].append(span)

    if "pain" in text.lower() and "pain" not in categories["Symptoms"]:
        categories["Symptoms"].append("pain")

    return categories

# Example usage
sample_text = """
Patient: I had a car accident on September 1st. I had neck and back pain.
Doctor: Diagnosis was whiplash injury. I had ten physiotherapy sessions.
"""
print(extract_medical_entities(sample_text))


🧪 Handling Missing or Ambiguous Data


Missing data is labeled as "Unknown" in the JSON output.


Low-confidence extractions can be flagged for manual review.


Hybrid models (rules + ML) reduce false positives.



🧠 Training Datasets (for fine-tuning)
DatasetDescriptionMTSamplesReal medical transcription examplesi2b2 Clinical NotesAnnotated clinical textsMedDialogDoctor–patient dialoguesHealthcare Sentiment DatasetEmotional tone classification

⚙️ Installation (Google Colab)
Run these commands at the top of your Colab notebook:
!pip install spacy transformers torch nltk textblob scikit-learn
!python -m spacy download en_core_web_sm


▶️ How to Use


Open the notebook Ashwini_Physician_Notetaker.ipynb in Google Colab.


Run all setup cells to install dependencies.


Paste or upload a physician-patient transcript.


The notebook will:


Extract medical entities


Analyze sentiment and intent


Generate a SOAP note





💡 Future Improvements


Fine-tune models using BioBERT / ClinicalBERT


Add confidence scores for each prediction


Deploy as a Streamlit / Gradio app for clinical users



👩‍💻 Author
Ashwini Umande
📧 ashwiniumande02@gmail.com
🗓️ Project Date: 07 November 2025

