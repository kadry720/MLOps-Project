from prefect import flow, task


@task
def validate_data():
    pass


@task
def preprocess():
    pass


@task
def train():
    pass


@task
def evaluate():
    pass


@task
def register_model():
    pass


@flow(name="mlops-training-pipeline")
def mlops_training_pipeline():
    validate_data()
    preprocess()
    train()
    evaluate()
    register_model()


if __name__ == "__main__":
    mlops_training_pipeline()
