# Support Ticket Use Case

This project demonstrates **support ticket classification** using multiple machine learning and deep learning approaches.

## Dataset

Support tickets are taken from the [support_ticket dataset](https://huggingface.co/datasets/phi-ai-info/support_tickets) on Hugging Face.

- **Source:** Hugging Face repo or JSON files in `tickets_dataset/`
- **Size:** 60 samples
- **Fields:**
  - `subject` — Short title for the ticket
  - `description` — Extended description of the issue
  - `key_phrase` — Key phrase indicating the ticket category

### Classes & Key Phrases

| Class   | Key Phrases |
|---------|-------------|
| access  | grant access, revoke access, access profile |
| user    | add user, delete user, modify user, create user |
| disk    | disk space, disk error, disk full |

## Notebooks

| Notebook | Description |
|----------|-------------|
| [support_tickets_problem.ipynb](support_tickets_problem.ipynb) | End-to-end pipeline: load data, embeddings, K-means clustering, n-grams, Logistic Regression, and DNN |
| [support_tickets_classification.ipynb](support_tickets_classification.ipynb) | Classification pipeline with [video walkthrough](https://www.youtube.com/watch?v=NlJ1yS0F03M) |
| [support_tickets_finetune_bert_classifier.ipynb](support_tickets_finetune_bert_classifier.ipynb) | Fine-tune BERT Base for support ticket classification |
| [support_tickets_finetune_bert_classifier_extended.ipynb](support_tickets_finetune_bert_classifier_extended.ipynb) | Extended BERT fine-tuning |
| [Pre-training and Fine-tuning.ipynb](Pre-training%20and%20Fine-tuning.ipynb) | Pre-training and fine-tuning concepts |

## Pipeline (support_tickets_problem.ipynb)

1. Load data
2. Preprocess data
3. Create embeddings for ticket descriptions
4. Perform dimensionality reduction
5. Analyze ticket descriptions with K-means clustering
6. Create 2-grams (word pairs) from descriptions
7. Create embeddings for word pairs
8. Analyze word pairs with K-means clustering
9. Generate generic labels for each ticket
10. Train and evaluate **Logistic Regression** (Scikit-learn)
11. Train and evaluate **Deep Neural Network** (Keras)

## Dependencies

- pandas
- matplotlib
- scikit-learn
- keras / tensorflow
- transformers
- datasets
- evaluate

## Project Structure

```
support_ticket_analysis/
├── tickets_dataset/          # JSON ticket data
├── support_tickets_problem.ipynb
├── support_tickets_classification.ipynb
├── support_tickets_finetune_bert_classifier.ipynb
├── support_tickets_finetune_bert_classifier_extended.ipynb
├── Pre-training and Fine-tuning.ipynb
├── best_model.h5             # Saved Keras model
└── Support Ticket Use Case slides.pdf
```

## Resources

- [Support Tickets dataset on Hugging Face](https://huggingface.co/datasets/phi-ai-info/support_tickets)
- [Support Tickets Automation: clustering and classification (YouTube)](https://www.youtube.com/watch?v=NlJ1yS0F03M)
