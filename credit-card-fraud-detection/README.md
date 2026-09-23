💳 Credit Card Fraud Detection

A machine learning project focused on detecting fraudulent credit card transactions using anonymized transaction data. The project explores multiple classification and clustering approaches to identify suspicious activity.

📌 Project Overview

Credit card fraud detection is a classic imbalanced-classification problem — fraudulent transactions are a small fraction of all transactions, which makes them hard to catch without also flagging too many legitimate ones. This project explores different machine learning approaches (both supervised and unsupervised) to identify suspicious transactions from anonymized transaction data.

📊 Dataset

The project uses an anonymized credit card transaction dataset containing transaction features and fraud labels (fraud vs. non-fraud).

🛠️ Models Used
K-Means Clustering — unsupervised approach to group transactions and flag outliers as potential fraud
Logistic Regression — baseline supervised classification model
Neural Network — deep learning approach for capturing non-linear patterns in the data
🧰 Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Jupyter Notebook
🔍 My Work
Performed data loading and preprocessing
Explored transaction data (EDA)
Implemented multiple machine learning models (K-Means, Logistic Regression, Neural Network)
Compared model outputs across approaches
Analysed fraud detection results
🚀 How to Run
bash
# Clone the repo
git clone https://github.com/SwatiPandita/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Install dependencies
pip install pandas numpy scikit-learn matplotlib jupyter

# Launch the notebook
jupyter notebook
🔮 Future Improvements
Add quantitative evaluation metrics (precision, recall, F1-score, ROC-AUC) to compare models more rigorously
Handle class imbalance explicitly (e.g. SMOTE or undersampling)
Deploy the best-performing model as a simple web app for real-time predictions
📄 License

This project is open source and available under the MIT License.
