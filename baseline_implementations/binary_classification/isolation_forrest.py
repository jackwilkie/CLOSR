from sklearn.ensemble import IsolationForest
import numpy as np
from numpy import ndarray

from baseline_implementations.common.model_eval import model_eval
from util.metrics import mean_auroc


class IsolationForestModel:
    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=1,
        )

    def fit(self, x: ndarray):
        self.model.fit(x)
        return self

    def predict(self, x: ndarray) -> ndarray:
        y_pred = self.model.predict(x)

        # sklearn IsolationForest returns:
        #   1  = inlier / benign
        #  -1  = outlier / malicious
        #
        # Convert to:
        #   0 = benign
        #   1 = malicious
        y_pred = np.where(y_pred == 1, 0, 1)

        return y_pred

    def fit_predict(
        self,
        x_train: ndarray,
        x_val: ndarray,
    ) -> ndarray:
        self.fit(x_train)
        return self.predict(x_val)

    def decision_function(self, x: ndarray) -> ndarray:
        return self.model.decision_function(x)

    def __call__(self, x: ndarray) -> ndarray:
        return self.predict(x)


def train_isolation_forest(model, x_train, y_train=None):
    if y_train is not None:
        y_train = y_train.copy()
        y_train[y_train > 0] = 1
        x_train = x_train[y_train == 0]

    model.fit(x_train)
    return model


def eval_isolation_forest(
    x_test,
    y_test,
    x_train,
    y_train,
    contamination: float = 0.1,
    n_estimators: int = 100,
    random_state: int = 42,
) -> dict:
    y_test = y_test.copy()
    y_train = y_train.copy()

    y_test[y_test > 1] = 1
    y_train[y_train > 1] = 1

    model = IsolationForestModel(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
    )

    # Train only on benign samples
    x_train_ = x_train[y_train == 0]
    model.fit(x_train_)

    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    train_results = model_eval(
        y_train,
        y_pred_train,
        label="train",
        return_class_level=True,
        return_detection_metrics=True,
    )

    test_results = model_eval(
        y_test,
        y_pred_test,
        label="test",
        return_class_level=True,
        return_detection_metrics=True,
    )

    return {**train_results, **test_results}


def fit_isolation_forest(
    x_train,
    x_val,
    y_val,
    y_train=None,
    contamination_vals=[
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
    ],
    n_estimators: int = 100,
    random_state: int = 42,
    label_prefix=None,
    val_type="move",
):
    label = label_prefix if label_prefix is not None else ""
    results = []

    y_val = y_val.copy()
    y_val[y_val > 0] = 1

    if y_train is not None:
        y_train = y_train.copy()
        y_train[y_train > 0] = 1

        if val_type == "move":
            i_move = np.where(y_train != 0)[0]

            x_val = np.concatenate((x_val, x_train[i_move]))
            y_val = np.concatenate((y_val, y_train[i_move]))

            x_train = np.delete(x_train, i_move, axis=0)
            y_train = np.delete(y_train, i_move, axis=0)
        else:
            x_train = x_train[y_train == 0]

    for contamination in contamination_vals:
        model = IsolationForestModel(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
        )

        y_pred = model.fit_predict(x_train, x_val)

        results_dict = model_eval(
            y_val,
            y_pred,
            label=label,
            return_class_level=True,
            return_detection_metrics=True,
        )

        results_dict["contamination"] = contamination
        results.append(results_dict)

    return results


def fit_isolation_forest_auroc(
    x_train,
    x_val,
    y_val,
    y_train=None,
    contamination_vals=[
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
    ],
    n_estimators: int = 100,
    random_state: int = 42,
    label_prefix=None,
    val_type="move",
):
    results = []

    y_val = y_val.copy()
    y_val[y_val > 0] = 1

    if y_train is not None:
        y_train = y_train.copy()
        y_train[y_train > 0] = 1

        if val_type == "move":
            i_move = np.where(y_train != 0)[0]

            x_val = np.concatenate((x_val, x_train[i_move]))
            y_val = np.concatenate((y_val, y_train[i_move]))

            x_train = np.delete(x_train, i_move, axis=0)
            y_train = np.delete(y_train, i_move, axis=0)
        else:
            x_train = x_train[y_train == 0]

    for contamination in contamination_vals:
        model = IsolationForestModel(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
        )

        model.fit(x_train)

        # Higher decision_function scores mean more normal.
        # Convert to anomaly score so higher = more malicious.
        scores = model.decision_function(x_val)
        anomaly_scores = 1 - scores

        results_dict = {"contamination": contamination}
        results_dict["auroc"] = mean_auroc(
            scores=anomaly_scores,
            y_true=y_val,
            return_class_level=False,
        )

        results.append(results_dict)

    return results
