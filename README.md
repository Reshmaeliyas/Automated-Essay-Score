# Automated-Essay-Score
# 📝 Automated Essay Scoring using Deep Learning

Welcome to the **Automated Essay Scoring** project — a machine learning solution that predicts the quality score of essays based on their textual content. This project leverages deep learning (LSTM/GRU) to evaluate essays, trained on a custom dataset and deployed using **Streamlit**.
## 🚀 Demo

🎯 Try it live: *(If deployed via ngrok or Streamlit Share, link goes here)*  
Example: `https://your-ngrok-or-streamlit-link`

---

## 📂 Project Structure

📦 Automated-Essay-Scoring
├── data/
│ └── automated_essay_score.csv
├── models/
│ ├── gru_essay_model.h5
│ └── tokenizer.pkl
├── streamlit_app.py
├── model_training.ipynb
└── README.md

markdown
Copy code

---

## 🧠 Model Overview

We trained a deep learning model using:
- **GRU (Gated Recurrent Units)** / LSTM
- **Text Preprocessing**: Lowercasing, tokenization, stopword removal
- **Embedding Layer**: Converts tokens into vector representations
- **SpatialDropout + GRU/LSTM**: Captures sequential dependencies in text
- **Loss Function**: Mean Squared Error (MSE)
- **Evaluation Metric**: Mean Absolute Error (MAE)

---

## 📊 Dataset

- The dataset consists of essays and their corresponding **manual scores**.
- Preprocessed and split into training and validation sets.

| Column       | Description            |
|--------------|------------------------|
| `full_text`  | Essay written by student |
| `score`      | Human-assigned score   |

---

## 🔧 Tech Stack

- `Python`
- `TensorFlow / Keras` (GRU model)
- `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`
- `Scikit-learn`
- `NLTK` for text cleaning
- `Streamlit` for front-end deployment
- `joblib` for model serialization
- `ngrok` for Colab deployment (optional)

---

## 🧪 How to Run Locally

1. **Clone the repo**:

```bash
git clone https://github.com/your-username/automated-essay-scoring.git
cd automated-essay-scoring
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy code
streamlit run streamlit_app.py
If you're using Google Colab, use pyngrok to expose the app.
The output by using the streamlit is attached below.
![Streamlit page](https://github.com/user-attachments/assets/8b610a9b-dbf7-4c25-bb6c-d0319384657c)


💾 Model Files
gru_essay_model.h5: Pre-trained GRU model

tokenizer.pkl: Fitted tokenizer used for text to sequence transformation

Make sure these files are in the same folder as streamlit_app.py for smooth loading.

📈 Results
Trained on cleaned essay data

Achieved low Mean Absolute Error (MAE) and consistent validation performance

Includes visualization of training loss curves

📌 Future Improvements
Add feedback explanation alongside score

Support for multiple languages

Deploy on Streamlit Cloud or HuggingFace Spaces

🧑‍💻 Author
Reshma
Aspiring Data Scientist | Passionate about NLP and Deep Learning
LinkedIn Profile (Update this link)



