CONSUMER COMPLAINT ANALYSIS

This project uses machine learning to predict whether a consumer will dispute a complaint filed with the Consumer Financial Protection Bureau (CFPB).

**Dataset:** CFPB Consumer Complaint Database

**Model:** LSTM network with Word2Vec embeddings.

**Steps:**

1. Data cleaning and preprocessing: Handling missing values, converting data types, and applying NLP techniques (tokenization, stemming, etc.).
2. Handling class imbalance using RandomOverSampler.
3. Word2Vec embedding for text data.
4. Building and training an LSTM model.
5. Evaluating model performance using accuracy and other metrics.
6. Deploying the model using Streamlit for real-time predictions.

**How to Run:**

1. Clone the repository.
2. Install required libraries: `pip install -r requirements.txt`.
3. Run the Colab Notebook: `colab notebook Consumer_Complaint_Analysis.ipynb`.
4.Run the Streamlit app: “ ! pip install streamlit -q “ , 
“`streamlit run app.py`” ..

**Note:**

- The dataset is assumed to be located at `/content/drive/MyDrive/DATA/CONSUMER COMPAINT ANALYSIS/complaints.csv`. Please update the path if necessary.
- Make sure you have necessary authentication for Google Drive if you intend to use it in Colab.

