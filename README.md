# Pegasus-Spyware-Information-Security-
### Multi-Model Analysis with Feature Engineering and Ensemble Methods

---

## Overview

Pegasus spyware is one of the most advanced surveillance tools, capable of infecting devices through zero-click attacks and remaining undetected using traditional security methods. Signature-based detection techniques often fail due to its evolving and stealthy nature.

This project presents a comprehensive machine learning approach to detect Pegasus spyware using behavioral indicators of compromise (IoCs). The study covers the full pipeline from data preprocessing and feature engineering to classical models, deep learning architectures, and advanced ensemble techniques.

---

## Objectives

- Detect Pegasus spyware using machine learning models
- Compare classical ML, deep learning, and ensemble approaches
- Identify and resolve data leakage issues
- Improve detection accuracy through feature engineering
- Build a production-ready detection strategy

---

## Project Structure

```
Pegasus-Spyware-Detection/
│
├── IS Final.py
├── Is_final_report.pdf
├── Information Security Final ppt.pptx
└── README.md
```

---

## Dataset

- Source: Synthetic Pegasus Spyware Dataset (Kaggle) :contentReference[oaicite:0]{index=0}  
- Total Samples: 1,000  
- Features: 17 (before feature engineering)  
- Classes:
  - None
  - Malicious IP
  - Pegasus Signature  

The dataset represents realistic behavioral patterns including system logs, network traffic, and device activity.

---

## Key Challenge: Data Leakage

Initial results showed:
- Training Accuracy: 100%
- Test Accuracy: ~33%

Root cause:
- Presence of high-cardinality features such as:
  - user_id
  - source_ip
  - destination_ip
  - timestamp

### Solution

- Removed leakage-prone features
- Engineered meaningful features:
  - Log-transformed data volume
  - Encryption flags
  - Error indicators
  - OS version grouping
- Grouped rare categories to reduce overfitting

Result:
- Stable generalization with consistent train/test performance

---

## Feature Engineering

Three techniques were evaluated:

| Method              | Accuracy | Improvement |
|--------------------|--------|------------|
| Baseline           | 84.3%  | —          |
| PCA                | 89.3%  | +5.0%      |
| Chi-Square         | 91.2%  | +6.9%      |
| Mutual Information | 90.1%  | +5.8%      |

### Key Insight
Feature engineering contributed more to performance than model complexity.

---

## Models

### Classical Machine Learning (6 Models)

- Logistic Regression
- Decision Tree
- Random Forest (Best Classical Model)
- K-Nearest Neighbors
- Support Vector Machine
- Naive Bayes

**Best Performance:**
- Random Forest: ~90.4% accuracy

---

### Deep Learning (3 Models)

- Multi-Layer Perceptron (MLP) — Best DL Model
- 1D Convolutional Neural Network (CNN)
- Long Short-Term Memory (LSTM)

**Best Performance:**
- MLP: ~92.7% accuracy

---

### Ensemble Methods (5 Types)

- Feature-Based Ensemble
- Data-Based Ensemble (Bagging)
- Model-Based Ensemble (Stacking)
- Model-Instance Ensemble
- Output-Based Ensemble (Voting)

**Best Overall Model:**
- Soft Voting Ensemble: ~95% accuracy

---

## Results Summary

| Model Type         | Best Model           | Accuracy |
|------------------|---------------------|--------|
| Classical ML      | Random Forest       | 90.4%  |
| Deep Learning     | MLP                 | 92.7%  |
| Ensemble Methods  | Soft Voting         | 95.0%  |

---

## Key Insights

- Feature engineering has greater impact than model selection  
- Ensemble models outperform individual models  
- Random Forest provides best balance of speed and accuracy  
- Deep learning gives slight improvement but requires more computation  
- Data leakage can severely mislead model performance  

---

## Technologies Used

- Python  
- Scikit-learn  
- TensorFlow / Keras  
- Pandas, NumPy  
- Matplotlib, Seaborn  

---

## Challenges

- Data leakage due to high-cardinality features  
- Overfitting in initial models  
- Handling categorical variables  
- Model instability before feature engineering  
- Balancing performance and computational cost  

---

## Future Work

- Use real-world spyware datasets  
- Implement real-time intrusion detection system  
- Integrate streaming data pipelines  
- Deploy lightweight models for edge devices  
- Improve explainability of predictions  

---

## Applications

- Cybersecurity intrusion detection systems  
- Mobile device threat monitoring  
- Network anomaly detection  
- Enterprise security analytics  

---

## Conclusion

This project demonstrates that:

Feature engineering and proper data handling are more important than model complexity in cybersecurity applications.

A combination of:
- Strong feature engineering  
- Classical machine learning  
- Ensemble techniques  

leads to robust and production-ready spyware detection systems.

---

## Author

Amrutha Gowri Jayasimha Hanumesh  
Master’s Student, Computer Science and Engineering  
Texas A&M University–San Antonio  
Course: Information Security
