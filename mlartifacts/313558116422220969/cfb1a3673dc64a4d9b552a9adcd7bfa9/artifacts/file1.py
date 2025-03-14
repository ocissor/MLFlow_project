import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import logging
import os

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)


# logging configuration
logger = logging.getLogger('file_1')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'file_1.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

mlflow.set_tracking_uri("http://localhost:5000")

logger.info("loading the wine dataset")
wine = load_wine()
X = wine.data
y = wine.target

logger.info("splitting the data into train and test sets")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=load_params('params.yaml')['random_forest_params']['random_state'])
logger.info("data split completed")

logger.info("initializing parameters for Random Forest")
max_depth = load_params('params.yaml')['random_forest_params']['max_depth']
n_estimators = load_params('params.yaml')['random_forest_params']['n_estimators']
random_state = load_params('params.yaml')['random_forest_params']['random_state']
logger.info("parameters initialized")

logger.info("MLFlow run started")

mlflow.set_experiment('exp1')
with mlflow.start_run():
    logger.info("defining the model")
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators,random_state=random_state)
    logger.info("model defined")

    logger.info("training the model")
    rf.fit(X_train, y_train)
    logger.info("model trained")

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    mlflow.log_metric('accuracy', accuracy)
    logger.info("accuracy logged")

    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    print(f"Accuracy is : {accuracy}")
    print(f'Confusion Matrix: {cm}')

    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    #save plot
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')
    mlflow.log_artifact(__file__)

    mlflow.set_tags({'Author': 'Sarthak', 'version': 'v1'})

    mlflow.sklearn.log_model(rf, 'random_forest_model')
    logger.info("model logged")


