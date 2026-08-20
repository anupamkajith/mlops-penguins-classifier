# ML Model Containerization using Docker

Name: Anupam K Ajith  
Roll Number: 2025EMAI10010  
Course: DSC524 Designing MLOps for Enterprises  
Course Duration: Jan–June 2026  

## Project Description
This project demonstrates how to containerize a Machine Learning experiment using Docker.  
The experiment trains a Decision Tree model using the Palmer Penguins dataset and outputs evaluation metrics.

## Project Structure
```
mlops-docker-ml-assignment-2025EMAI10010
│
├── README.md
├── .gitignore
└── ml_experiment
    ├── palmer-panguin-decisionTree.py
    ├── requirements.txt
    └── Dockerfile
```
## How to Run

Build Docker image:

docker build -t penguin-ml-model .

Run container:

docker run penguin-ml-model
