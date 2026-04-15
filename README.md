# AI Skill Competency Evaluation System (Final Project)

## 1. Overview

This project develops a prototype system that evaluates employee skill competency using Natural Language Processing (NLP) and data-driven scoring techniques. The system processes employee data (e.g., job history and listed skills) and produces a **quantitative competency score per skill**, supported by textual evidence.

The primary objective is to address the lack of reliable skill visibility within organisations, which can lead to inefficient resourcing and over-reliance on external contractors.

---

## 2. Objectives

* Extract relevant skill-related information from employee text data
* Generate a **competency score per skill** using an Automated Essay Scoring (AES)-inspired approach
* Provide a transparent and explainable scoring mechanism
* Evaluate model outputs against expected or perceived skill levels
* Deliver a working MVP demonstrating feasibility within real-world constraints

---

## 3. System Architecture (High-Level)

**Pipeline Flow:**

1. Data Ingestion
2. Data Validation
3. Data Preprocessing
4. Feature Engineering (TF-IDF)
5. Similarity Calculation (Cosine Similarity)
6. Competency Scoring (AES Logic)
7. Output Generation

---

## 4. Methodology

### 4.1 Data Ingestion

* Input format: CSV
* Key fields:

  * `Job History`
  * `Skills`

### 4.2 Data Preprocessing

* Implemented using **spaCy**
* Steps:

  * Tokenisation
  * Stopword removal
  * Text cleaning

### 4.3 Feature Engineering

* TF-IDF is used to convert text into numerical vectors
* Highlights **important and rare skill-related terms**

### 4.4 Similarity Measurement

* Cosine similarity is applied to:

  * Compare employee text vs reference skill descriptions
* Produces a similarity score representing **alignment with ideal skill profiles**

### 4.5 Competency Scoring (Core Logic)

* AES-inspired approach:

  * Uses similarity scores as input
  * Outputs a **competency score per skill**
* Optional:

  * Neural network (BPNN) for score prediction

---

## 5. Technologies Used

* Python
* Pandas (data handling)
* spaCy (NLP preprocessing)
* Scikit-learn (TF-IDF, cosine similarity)
* Pytest (data validation/testing)

---

## 6. Project Structure

```
project/
│── data/
│   └── talentlink_data.csv
│
│── src/
│   ├── ingestion.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── scoring.py
│   └── evaluation.py
│
│── tests/
│   └── test_data_validation.py
│
│── requirements.txt
│── README.md
```

---

## 7. Evaluation Strategy

* Compare predicted competency scores with:

  * Employee expectations (or proxy validation)
* Metrics:

  * Accuracy threshold (target ≥ 80%)
  * Similarity alignment

---

## 8. Limitations

* Small dataset (~100 employees)
* No real-world deployment validation
* Reliance on predefined “ideal” skill descriptions
* Limited bias mitigation

---

## 9. Future Improvements

* Integrate **skill taxonomy** for better skill relationships
* Use **LLMs** for explainability and evidence generation
* Implement **knowledge graphs** for skill inference
* Deploy system on **cloud infrastructure (AWS/Azure)**
* Expand dataset for improved model reliability

---

## 10. How to Run

### Install dependencies

```
pip install -r requirements.txt
```

### Run pipeline

```
python src/main.py
```

### Run tests

```
pytest
```

---

## 11. Expected Output

* Table or structured output:

  * Employee ID
  * Skill
  * Competency Score
  * Supporting Evidence

---

## 12. Author

Final Year Project – Queen Mary University of London
Degree Apprenticeship (Digital & Technology Solutions – Data Analyst)

---
