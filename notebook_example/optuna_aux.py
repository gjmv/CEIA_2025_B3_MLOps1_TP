import mlflow 

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


def champion_callback(study, frozen_trial):
    """
    Callback para Optuna que imprime un mensaje cuando un nuevo trial supera 
    el mejor valor actual (best_value). También imprime cada 10 trials como progreso.

    Parámetros:
    -----------
    study : optuna.study.Study
        El objeto Study actual que está ejecutando la optimización.
    frozen_trial : optuna.trial.FrozenTrial
        Trial finalizado que acaba de completarse.
    """
    
    # Obtener el mejor valor previo registrado en user_attrs
    winner = study.user_attrs.get("winner", None)
    
    # Si el valor actual supera al ganador anterior
    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        
        if winner:
            # Calcular la mejora porcentual
            improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
            print(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
        else:
            # Primer ganador registrado
            print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")
    elif frozen_trial.number % 10 == 0:
        # Mostrar progreso cada 10 trials
        print(f"Trial {frozen_trial.number} with no changes.")

def objective(trial, X_train, y_train, experiment_id):
    """
    Función objetivo para la optimización de hiperparámetros con Optuna.

    Esta función entrena un clasificador `DecisionTreeClassifier` con distintos 
    valores de hiperparámetros sugeridos por Optuna. Registra cada ejecución 
    como un "run" anidado en MLflow, guarda los parámetros y el F1 Score.

    Parámetros:
    -----------
    trial : optuna.trial.Trial
        Objeto que representa una iteración de búsqueda de hiperparámetros.
    
    X_train : pandas.DataFrame
        Conjunto de datos de entrada para entrenamiento.
    
    y_train : pandas.Series
        Etiquetas del conjunto de entrenamiento.
    
    experiment_id : int
        ID del experimento de MLflow donde se registrarán los resultados.

    Retorna:
    --------
    float
        Promedio del F1 Score ponderado en validación cruzada.
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
