# t5QG-model-development

# T5 Question Generation Finetuning Development with Hyperparameter Optimization

## Overview
This project implements automatic question generation using T5-base transformer model with hyperparameter optimization (HPO) using Optuna. The system takes a context passage and an answer as input, then generates relevant questions.

## Features
- **Data Preprocessing**: Text cleaning, normalization, and stratified dataset splitting
- **Hyperparameter Optimization**: Automated HPO using Optuna with TPE sampler and GridSearch
- **Model Fine-tuning**: T5-base fine-tuning with optimal hyperparameters
- **Comprehensive Evaluation**: BLEU, METEOR, and ROUGE-L metrics
- **Dropout Rate Optimization**: Optional dropout rate tuning for better generalization

## Requirements
```
torch
transformers>=4.41.0
pytorch_lightning
optuna
nltk==3.8.1
sacrebleu
rouge-score
scikit-learn
matplotlib
seaborn
pandas
numpy
```

## Dataset Format
The input dataset should contain:
- `context`: The passage text
- `answers`: The target answer
- `question`: The ground truth question
- `question_type`: Type of question (optional for stratified splitting)

## Project Structure
```
project/
├── dataset/           # Processed datasets
├── model/            # Saved fine-tuned models
├── tokenizer/        # Saved tokenizers
├── log/             # Training logs and HPO results
├── evaluation/      # Evaluation results and plots
└── optuna/          # Optuna study results and visualizations
```

## Usage

### 1. Data Preprocessing
```python
# Load and clean dataset
df = pd.read_excel("dataset_final.xlsx")
# Text normalization and cleaning
# Stratified train/val/test split (80/10/10)
```

### 2. Hyperparameter Optimization
```python
# Define search space
search_space = {
    "learning_rate": [2e-5, 3e-5, 4e-5, 5e-5],
    "batch_size": [4, 6, 8, 12],
    "weight_decay": [1e-6, 1e-5, 5e-5, 1e-4],
    "dropout_rate": [0.1, 0.125, 0.15]  # Optional
}

# Run optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
```

### 3. Model Fine-tuning
```python
# Load optimal hyperparameters
best_params = study.best_params

# Fine-tune T5-base model
model = T5ForConditionalGeneration.from_pretrained("t5-base")
# Training with early stopping
```

### 4. Evaluation
```python
# Generate questions on test set
# Calculate BLEU, METEOR, ROUGE-L scores
# Save results and visualizations
```

## Key Results
The HPO process explores different configurations to find optimal:
- Learning rate: Typically 3e-5 to 5e-5
- Batch size: 6-8 for best performance
- Weight decay: 1e-5 to 1e-4
- Dropout rate: 0.10-0.15 (when enabled)

## Evaluation Metrics
- **BLEU-1/2/3/4**: N-gram precision scores
- **METEOR**: Considers synonyms and paraphrases
- **ROUGE-L**: Longest common subsequence
- **SacreBLEU**: Standardized BLEU implementation

## Output Examples
```
Context: "Sushi is a special food from Japan..."
Answer: "Japan"
Generated Question: "Where does sushi come from?"
```

## Files Generated
- Model checkpoints and tokenizers
- Training loss plots
- HPO analysis reports and visualizations
- Evaluation results with metric correlations
- Generated question samples

## Notes
- Supports both TPE sampler and GridSearch for HPO
- Implements early stopping based on validation loss
- Includes comprehensive logging and monitoring
- Generates detailed analysis reports for HPO results