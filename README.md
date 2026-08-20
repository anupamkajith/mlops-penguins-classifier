# Dockerized Palmer Penguins Classifier

A machine learning classification project demonstrating how to package and run a scikit-learn model inside a Docker container.

The project was originally developed as part of the **DSC524 – Designing MLOps for Enterprises** coursework in the M.Tech Artificial Intelligence and Data Science program at IIIT Kottayam.

---

## 🎯 Project Objective

The objective of this project is to demonstrate a basic MLOps workflow by combining:

- Machine learning model development
- Data preprocessing
- Model evaluation
- Reproducible environments
- Docker containerization

The model predicts the species of a penguin using physical and categorical characteristics from the Palmer Penguins dataset.

---

## 🐧 Dataset

The project uses the **Palmer Penguins dataset**, which contains measurements for three penguin species:

- Adelie
- Chinstrap
- Gentoo

Features include:

- Island
- Bill length
- Bill depth
- Flipper length
- Body mass
- Sex
- Year

Rows containing missing values are removed before training.

Dataset reference:

[Palmer Penguins](https://github.com/allisonhorst/palmerpenguins)

---

## 🛠 Tech Stack

- Python
- pandas
- scikit-learn
- Palmer Penguins
- Docker

---

## 🔄 ML Workflow

```text
Palmer Penguins Dataset
        ↓
Data Cleaning
        ↓
Feature / Target Split
        ↓
Train-Test Split
        ↓
Categorical Encoding
        ↓
Decision Tree Classifier
        ↓
Model Evaluation
        ↓
Docker Container
