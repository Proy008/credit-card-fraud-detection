# Credit Card Fraud Detection

**Author:** Probal Roy  
**College:** IISER Thiruvananthapuram

---

It is important that credit card companies are able to recognise fraudulent credit card transactions so that customers are not charged for items that they did not purchase.

The dataset contains transactions made by credit cards in September 2013 by European cardholders. It presents transactions that occurred over two days, where 492 out of 284,807 transactions are fraudulent. The dataset is highly imbalanced — the positive class (frauds) accounts for only 0.172% of all transactions.

Download the dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud/).

---

## Notebooks

| Notebook | Description |
|---|---|
| `fraud_detection_anomaly_models.ipynb` | Unsupervised anomaly detection using Isolation Forest, LOF, and One-Class SVM |
| `fraud_detection_knn_classifier.ipynb` | Supervised KNN classifier with cross-validated K selection |

---

## Model Approaches

### Isolation Forest Algorithm
One of the newer techniques for anomaly detection. It is based on the observation that anomalies are data points that are few and different — properties that make them susceptible to *isolation*.

The algorithm isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum of that feature. Isolating anomalous observations requires fewer splits than isolating normal ones, so an anomaly score can be derived from the path length in a random decision tree. The method has low linear time complexity and a small memory footprint.

### Local Outlier Factor (LOF)
An unsupervised outlier detection method that computes the local density deviation of a data point relative to its neighbours. Points with substantially lower density than their neighbours are flagged as outliers.

The number of neighbours (`n_neighbors`) is typically chosen to be greater than the minimum cluster size but smaller than the number of nearby points that could be local outliers. In practice, `n_neighbors=20` works well as a default.

### K-Nearest Neighbors (KNN) Classifier
A supervised approach where the optimal value of K is selected by maximising cross-validated recall score — recall being the priority metric to minimise missed fraudulent transactions. The best K is then used to train a final classifier evaluated on a held-out test set.

---

## Results Summary (Anomaly Detection)

- **Isolation Forest** performed best overall, detecting fraud cases with ~27% precision at 99.74% accuracy.
- **Local Outlier Factor** achieved 99.65% accuracy but only ~2% fraud detection rate.
- **One-Class SVM** had the lowest performance at ~70% accuracy for this dataset.
