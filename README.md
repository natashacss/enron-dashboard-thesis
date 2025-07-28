# Enron Anomaly Detection Dashboard

This Streamlit-based dashboard visualizes anomaly detection results from the **Enron Email Dataset** using machine learning models. It is developed as part of an undergraduate thesis project focused on **Human Intelligence (HUMINT)-inspired analysis** of corporate communication patterns.

---

## Project Overview

This system applies three unsupervised anomaly detection models to identify potentially suspicious or abnormal email behavior:

- **Isolation Forest**
- **Local Outlier Factor (LOF)**
- **One-Class SVM**

The results are combined and visualized to provide an interactive, explainable view into email-based anomalies across Enron employees.

---

## Features

- **Score Distribution Analysis**  
  Understand how each model evaluates normal vs anomalous behavior.

- **Model Agreement Metrics**  
  See how often models agree on flagged emails.

- **Communication Network Graph**  
  Visual map of sender-recipient interactions, colored by anomaly severity.

- **Early Warning System Table**  
  Randomized spotlight on high-risk individuals flagged by multiple models.

- **Interactive Search Tool**  
  Look up emails by sender and view their subjects & full messages.

---

## Tech Stack

- Python + Streamlit
- pandas, seaborn, scikit-learn
- networkx, pyvis
- Data: [Enron Email Dataset (Kaggle)](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset)

---

## Academic Context

This project was created as part of the undergraduate thesis:

> **"Design and Implementation of Human Resource Intelligence Applications based on Human Intelligence for Anomaly Detection"**

- **Name:** Natasha Catherine Siringoringo  
- **Student ID:** 18121056  
- **University:** Bandung Institute of Technology  
- **Telematics Lab**

---

## Contact

Feel free to reach out for academic inquiries, collaborations, or dashboard improvements!  
*katarinatashaa@gmail.com*
