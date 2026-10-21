# Suspicious Message Classification Using Machine Learning

## 📌 Project Overview

Suspicious Message Classification is a Machine Learning project that identifies whether an SMS message is **Spam (Suspicious)** or **Ham (Legitimate)**.

The project uses the SMS Spam Collection dataset from the UCI Machine Learning Repository. It applies text preprocessing and TF-IDF feature extraction to convert text messages into numerical features that machine learning algorithms can process.

A Flask-based web application allows users to enter a message and view its predicted classification.

## 🎯 Objectives

- Classify SMS messages as Spam or Ham.
- Apply text preprocessing techniques to clean message data.
- Convert text into numerical features using TF-IDF.
- Train and compare multiple machine learning algorithms.
- Evaluate models using standard classification metrics.
- Build a simple web interface for message prediction.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Web Framework:** Flask
- **Data Processing:** Pandas, NumPy
- **Feature Extraction:** TF-IDF Vectorizer
- **Model Saving:** Joblib
- **Frontend:** HTML, CSS, JavaScript
- **Dataset:** UCI SMS Spam Collection

## 🤖 Machine Learning Algorithms

The project compares the following algorithms:

1. Logistic Regression
2. Linear Support Vector Machine (Linear SVM)
3. K-Nearest Neighbors (KNN)
4. AdaBoost Classifier

The models are evaluated using accuracy, precision, recall, F1-score, confusion matrices, and classification reports. The best-performing model is selected based on the evaluation results.

## 📂 Project Structure

```text
Suspicious-Message-Classification/
│
├── app.py
├── requirements.txt
├── best_spam_classifier.pkl
├── model_comparison.csv
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── script.js
│
└── README.md
```

*Note: The saved model and comparison CSV may be generated when the application trains the models. The exact files may vary depending on your project.*

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/snehaspatil721-coder/Suspicious-Message-Classification.git
```

### 2. Open the Project Folder

```bash
cd Suspicious-Message-Classification
```

### 3. Create a Virtual Environment (Optional)

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
python app.py
```

### 6. Open the Website

Open your browser and visit:

http://127.0.0.1:5000

Enter a message in the input box and submit it to see the predicted result.

## 🔄 Project Workflow

1. Collect the SMS Spam Collection dataset.
2. Clean and preprocess the text messages.
3. Convert messages into TF-IDF features.
4. Train the machine learning models.
5. Compare the models using evaluation metrics.
6. Select the best-performing model.
7. Use Flask to connect the model to the web interface.
8. Display the prediction to the user.

## 📊 Evaluation Metrics

- **Accuracy:** Measures the proportion of correct predictions.
- **Precision:** Measures how many messages predicted as spam are actually spam.
- **Recall:** Measures how many actual spam messages are correctly identified.
- **F1-Score:** Combines precision and recall into one metric.
- **Confusion Matrix:** Shows correct and incorrect classifications.

## 🌐 Web Application Features

- Simple and user-friendly interface.
- Text input for SMS messages.
- Spam and Ham classification.
- Prediction result displayed on the webpage.
- Machine learning model integrated with a Flask backend.

## 📚 Dataset

The project uses the SMS Spam Collection dataset from the UCI Machine Learning Repository.

Dataset source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

## 🚀 Future Enhancements

- Deploy the application online.
- Improve model performance using additional text-processing techniques.
- Add more machine learning algorithms.
- Provide visualizations of model performance.
- Extend classification to other types of suspicious messages.

## 👩‍💻 Author

**Sneha Patil**

GitHub Repository: https://github.com/snehaspatil721-coder/Suspicious-Message-Classification

## 📄 Conclusion

This project demonstrates how machine learning and natural language processing techniques can be used to classify SMS messages as spam or legitimate. It combines data preprocessing, TF-IDF feature extraction, model evaluation, and Flask web development in one application.
