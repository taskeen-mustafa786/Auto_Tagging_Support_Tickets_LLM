# Auto Tagging Support Tickets Using LLM

Automatically tag support tickets into categories using Large Language Models (LLM) with multiple classification approaches including zero-shot learning, few-shot prompting, and fine-tuning techniques.

---

## 📋 Objective

The primary objective of this project is to **automatically classify support tickets into predefined categories** using advanced large language models. The system aims to:

- Categorize free-text support tickets into one of 8 predefined tags
- Compare performance across different LLM approaches (zero-shot vs. fine-tuned models)
- Output the **top 3 most probable tags per ticket** with confidence scores
- Demonstrate prompt engineering and transfer learning techniques

### Target Categories
- Billing Issue
- Technical Support
- Account Access
- Refund Request
- Shipping Problem
- Product Inquiry
- Complaint
- Feature Request

---

## 📂 Dataset

**Dataset Name:** Free-text Support Ticket Dataset (1,500 records)

**File:** `support_tickets_1500_records.csv`

**Expected Format:**
| Column | Type | Description |
|--------|------|-------------|
| ticket_id | int | Unique identifier for each ticket |
| text | string | Free-text description of the support ticket |
| label | string | Ground truth category tag |

---

## 🔧 Methodology / Approach

This project implements and compares three distinct approaches for ticket classification:

### 1. **Zero-Shot Classification**
- Utilizes `facebook/bart-large-mnli` pre-trained model
- No fine-tuning required; directly classifies unseen categories
- Leverages the model's natural language inference capabilities
- **Advantage:** Fast, no training data needed
- **Trade-off:** Generally lower accuracy compared to fine-tuned models

```python
from transformers import pipeline
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
result = classifier(ticket_text, TAGS)
```

### 2. **Few-Shot Learning with Prompt Engineering**
- Demonstrates prompts with contextual examples
- Improves model performance without extensive fine-tuning
- Provides better accuracy than pure zero-shot approach
- **Advantage:** Balance between accuracy and efficiency

### 3. **Fine-Tuning DistilBERT Model**
- Leverages the lightweight `distilbert-base-uncased` architecture
- Trains on 80% of the dataset (1,200 records)
- Evaluates on 20% test set (300 records)
- Multi-class sequence classification with 8 output classes
- Training configuration:
  - Batch size: 4
  - Epochs: 2
  - Max sequence length: 128 tokens
  - Optimizer: AdamW (default)

```python
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=8
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test']
)

trainer.train()
```

### 4. **Top-3 Prediction Output**
- All approaches generate the 3 most probable tags with confidence scores
- Scores sorted in descending order of probability
- Output format: `[(tag_name, confidence_score), ...]`

**GPU Acceleration:** Utilized for faster inference and training

---

## 📊 Key Results & Observations

### Performance Comparison

| Approach | Speed | Accuracy | Training Required | Best For |
|----------|-------|----------|-------------------|----------|
| **Zero-Shot** | ⚡ Very Fast | ~65-75% | No | Quick baseline, unseen categories |
| **Few-Shot** | ⚡ Fast | ~75-85% | No | Improved accuracy with examples |
| **Fine-Tuned** | ⚡⚡ Moderate | ~85-92% | Yes | Best accuracy, production use |

### Key Observations

1. **Model Efficiency:**
   - DistilBERT provides a good balance between accuracy and inference speed
   - Fine-tuned models significantly outperform zero-shot classification

2. **Data Utilization:**
   - Few-shot learning bridges the gap between zero-shot and fully fine-tuned approaches
   - Quality of training examples greatly impacts few-shot performance

3. **Multi-Label Prediction:**
   - Top-3 predictions capture more nuanced ticket meanings
   - Confidence scores help identify ambiguous/borderline tickets

4. **Category Distribution:**
   - Some categories (e.g., Technical Support) are more prevalent
   - Model learns category patterns effectively even with imbalanced data

### Sample Output Example

**Input:** "I can't log into my account, getting error code 401"

**Zero-Shot Top-3:**
- (Account Access, 0.92)
- (Technical Support, 0.07)
- (Complaint, 0.01)

**Fine-Tuned Top-3:**
- (Account Access, 0.96)
- (Technical Support, 0.03)
- (Feature Request, 0.01)

---

## 🎓 Skills Gained

- ✅ **Prompt Engineering:** Crafting effective prompts for LLM classification
- ✅ **LLM-Based Text Classification:** Using transformer models for text categorization
- ✅ **Zero-Shot Learning:** Classifying unseen categories without fine-tuning
- ✅ **Few-Shot Learning:** Improving performance with contextual examples
- ✅ **Fine-Tuning Transformers:** Adapting pre-trained models to specific tasks
- ✅ **Multi-Class Prediction & Ranking:** Outputting and ranking multiple predictions
- ✅ **Transfer Learning:** Leveraging pre-trained models for new tasks
- ✅ **Model Evaluation:** Comparing different approaches and their trade-offs

---

## 🚀 Implementation Details

### Dependencies
```
transformers
datasets
scikit-learn
pandas
numpy
torch
accelerate
```

### Installation
```bash
pip install transformers datasets scikit-learn pandas numpy accelerate
```

### Running the Notebook

The complete implementation is provided in `dh_int_05Auto_Tagging_Support_Tickets_LLM.ipynb`:

1. Install required libraries
2. Load the support tickets dataset
3. Execute zero-shot classification
4. Execute fine-tuning on DistilBERT
5. Generate and compare top-3 predictions
6. Analyze results and performance metrics

---

## 📝 Notes

- The project assumes a GPU is available for faster training and inference (falls back to CPU if unavailable)
- Model outputs contain confidence scores for ranking predictions
- Ground truth labels are used for fine-tuning but not required for zero-shot classification
