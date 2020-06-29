import click
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlxtend.evaluate import mcnemar_table
from mlxtend.evaluate import mcnemar


# We fix the random seed to make the experiment repeatable
random_seed = 16
np.random.seed(random_seed)


@click.command()
@click.option(
    "--mlflow-server",
    type=str,
    default="http://localhost:5000",
    help="Address of the MLFlow server",
)
@click.option(
    "--significance",
    type=float,
    default=0.05,
    help="Significance level for the McNemar test",
)
def main(
    mlflow_server: str, significance: float,
):
    # We start by setting the tracking uri to make sure the mlflow server is reachable
    mlflow.set_tracking_uri(mlflow_server)
    # We need to instantiate the MlflowClient class for certain operations
    mlflow_client = MlflowClient()
    # We create and set an experiment to group all runs
    mlflow.set_experiment("Model Comparison")

    # We create classification data and split it into training and testing sets
    X, y = make_classification(
        n_samples=10000,
        n_classes=2,
        n_features=20,
        n_informative=9,
        random_state=random_seed,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, test_size=0.2
    )

    # We first train a Logistic regression model, log it in mlflow and then move it to the production stage
    with mlflow.start_run():
        lr_model = LogisticRegression()
        lr_model.fit(X_train, y_train)
        y_pred = lr_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(
            lr_model, artifact_path="model", registered_model_name="Logistic Regression"
        )
    mlflow_client.transition_model_version_stage(
        name="Logistic Regression", version=1, stage="Production"
    )

    # We then train a Random Forest model, log it in mlflow and then move it to the staging stage
    with mlflow.start_run():
        rf_model = RandomForestClassifier()
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(
            lr_model, artifact_path="model", registered_model_name="Random Forest"
        )
    mlflow_client.transition_model_version_stage(
        name="Random Forest", version=1, stage="Staging"
    )

    # We finally load both models from MLFlow
    # and compare them using the McNemar test
    # We get the download uris of both models and then we load them
    lr_model_download_uri = mlflow_client.get_model_version_download_uri(
        name="Logistic Regression", version=1,
    )
    rf_model_download_uri = mlflow_client.get_model_version_download_uri(
        name="Random Forest", version=1,
    )
    lr_model = mlflow.sklearn.load_model(lr_model_download_uri)
    rf_model = mlflow.sklearn.load_model(rf_model_download_uri)

    y_pred_lr = lr_model.predict(X_test)
    y_pred_rf = rf_model.predict(X_test)

    contingency_table = mcnemar_table(y_test, y_pred_lr, y_pred_rf)
    _, p_value = mcnemar(contingency_table, corrected=True)

    if p_value < significance:
        # In this case we reject the null hypothesis that the two models' are similar
        # We then archive the logistic regression model
        # and move the random forest model to the Production stage
        print(f"p-value {p_value} smaller than significance level {significance}")
        print(
            "Archiving logistic regression model and moving random forest model to production"
        )
        mlflow_client.transition_model_version_stage(
            name="Logistic Regression", version=1, stage="Archived",
        )
        mlflow_client.transition_model_version_stage(
            name="Random Forest", version=1, stage="Production",
        )
    else:
        print(f"p-value {p_value} greater than significance level {significance}")
        print("Keeping logistic regression model in production")


if __name__ == "__main__":
    main()
