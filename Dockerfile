FROM python:3.8.3-slim-buster

RUN pip install mlflow==1.9.1

CMD ["mlflow", "server", "--host=0.0.0.0", \
     "--backend-store-uri=sqlite:///mlflow.db", \
     "--default-artifact-root=/tmp/mlflow"]
