#Use python base image
FROM python:3.10

#Working directory
WORKDIR /app

#copy project files
COPY requirements.txt .
COPY palmer-panguin-decisionTree.py .


#intsall dependencies
RUN pip install --no-cache-dir -r requirements.txt

#Run Ml experiment
cmd ["python","palmer-panguin-decisionTree.py"]
