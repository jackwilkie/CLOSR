from sklearn import svm
import numpy as np
from numpy import ndarray
from baseline_implementations.common.model_eval import model_eval
from util.metrics import mean_auroc


class OneClassSVM:
    def __init__(
        self,
        nu: float = 0.5,
    ):

        self.svm = svm.OneClassSVM(nu=nu, verbose=True)

    def fit(self, x: ndarray) -> ndarray:
        self.svm.fit(x)

    def predict(self, x: ndarray) -> ndarray:
        y_pred = self.svm.predict(x)
        y_pred = np.where(y_pred == 1, 1, 0)
        return y_pred

    def fit_predict(
        self,
        x_train,
        x_val,
    ):
        self.fit(x_train)
        return self.predict(x_val)

    def decision_function(self, x):
        return self.svm.decision_function(x)

    def __call__(self, x):
        return self.predict(x)


def train_svm(model, x_train, y_train=None):
    return x_train, y_train

    if y_train is not None:
        x_train = x_train[y_train == 0]
    model = model.fit(x_train)


def eval_svm(
    x_test,
    y_test,
    x_train,
    y_train,
    nu: float = 0.5,
) -> dict:
    y_test[y_test > 1] = 1
    y_train[y_train > 1] = 1

    model = OneClassSVM(nu)

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


def fit_svm(
    x_train,
    x_val,
    y_val,
    y_train=None,
    nu_vals=[
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
    ],
    label_prefix=None,
    val_type="move",
):
    label = label_prefix if label_prefix is not None else ""
    results = []
    y_val[y_val > 0] = 1

    if y_train is not None:
        y_train[y_train > 0] = 1
        if val_type == "move":
            i_move = np.where(y_train != 0)[0]

            x_val = np.concatenate((x_val, x_train[i_move]))
            y_val = np.concatenate((y_val, y_train[i_move]))

            x_train = np.delete(x_train, i_move, axis=0)
            y_train = np.delete(y_train, i_move, axis=0)
        else:
            x_train = x_train[y_train == 0]

    for nu in nu_vals:
        model = OneClassSVM(nu)
        y_pred = model.fit_predict(x_train, x_val)
        results_dict = model_eval(
            y_val,
            y_pred,
            label=label,
            return_class_level=True,
            return_detection_metrics=True,
        )
        results_dict["nu"] = nu
        results.append(results_dict)

    return results


def fit_svm_auroc(
    x_train,
    x_val,
    y_val,
    y_train=None,
    nu_vals=[
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
    ],
    label_prefix=None,
    val_type="move",
):
    results = []
    y_val[y_val > 0] = 1

    if y_train is not None:
        y_train[y_train > 0] = 1
        if val_type == "move":
            i_move = np.where(y_train != 0)[0]

            x_val = np.concatenate((x_val, x_train[i_move]))
            y_val = np.concatenate((y_val, y_train[i_move]))

            x_train = np.delete(x_train, i_move, axis=0)
            y_train = np.delete(y_train, i_move, axis=0)
        else:
            x_train = x_train[y_train == 0]

    for nu in nu_vals:
        model = OneClassSVM(nu)
        model.fit(x_train)
        scores = model.svm.decision_function(x_val)
        results_dict = {"nu": nu}
        results_dict["auroc"] = mean_auroc(
            scores=1 - scores, y_true=y_val, return_class_level=False
        )
        results.append(results_dict)

    return results
