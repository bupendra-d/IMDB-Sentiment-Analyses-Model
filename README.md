## IMDB Sentiment Analysis Model
This project is an end-to-end Natural Language Processing (NLP) solution that classifies IMDB movie reviews as **Positive** or **Negative** using machine learning techniques.
The pipeline combines **TF-IDF text vectorization** with **Logistic Regression**, and includes proper data cleaning, stratified splitting, model comparison, cross-validation, and inference.



## Project Highlights
- Text preprocessing using TF-IDF
- Stratified train-test split
- Model comparison (Logistic Regression, Decision Tree, Random Forest)
- Cross-validation for fair evaluation
- End-to-end pipeline using `sklearn.pipeline`
- Model & pipeline persistence using `joblib`
- Ready-to-use inference workflow



## Best Performing Model
Based on cross-validation accuracy, **Logistic Regression** was selected as the final model due to its strong performance and interpretability for text classification tasks.



## Project Structure
├── IMDB_dataset.csv
├── model_experiment.ipynb             # Model experimentation & comparison
├── model_building.py                  # Training & inference script
├── imdb_model.pkl                     # Saved model
├── imdb_pipeline.pkl                  # Saved TF-IDF + model pipeline
├── input.csv                          # Sample input for inference
├── output.csv                         # Predictions
├── requirements.txt
└── README.md



## How to Run
```bash
pip install -r requirements.txt
python model_building.py
