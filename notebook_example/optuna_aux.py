import mlflow 

from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def champion_callback(study, frozen_trial):
    """
    Logging callback that will report when a new trial iteration improves upon existing
    best trial values.
    """

    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
            print(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
        else:
            print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")
    elif frozen_trial.number % 10 == 0:
        print(f"Trial {frozen_trial.number} with no changes.")

def objective(trial, X_train, y_train, experiment_id):
    """
    Optimize hyperparameters for a classifier using Optuna.

    Parameters:
    -----------
    trial : optuna.trial.Trial
        A trial is a process of evaluating an objective function.
    X_train : pandas.DataFrame
        Input features for training.
    y_train : pandas.Series
        Target variable for training.
    experiment_id : int
        ID of the MLflow experiment where results will be logged.

    Returns:
    --------
    float
        Mean F1 score of the classifier after cross-validation.
    """

    # Comienza el run de MLflow. Este run debería ser el hijo del run padre, 
    # así se anidan los diferentes experimentos.
    with mlflow.start_run(experiment_id=experiment_id, 
                          run_name=f"Trial: {trial.number}", nested=True):

        max_depth = trial.suggest_int("max_depth", 1, 30)
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)

        classifier = DecisionTreeClassifier(criterion=criterion, splitter='best', 
                                            max_depth=max_depth, min_samples_split=min_samples_split, 
                                            min_samples_leaf=min_samples_leaf, random_state=42)

        # Parámetros a logguear
        params = {
            "eval_metric": "f1_weighted",
            "max_depth": max_depth,
            "criterion": criterion,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf
        }

        # Realizamos validación cruzada y calculamos el score F1
        score = cross_val_score(classifier, X_train, y_train.to_numpy().ravel(), 
                                n_jobs=-1, cv=5, scoring='f1_weighted')
        
        # Log los hiperparámetros a MLflow
        mlflow.log_params(params)
        # Y el score f1 medio de la validación cruzada.
        mlflow.log_metric("f1_weighted", score.mean())

    return score.mean()
