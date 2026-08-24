from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import itertools
import numpy as np
import torch
import torch.nn as nn

from .model import FusionNet


def make_loader(arrays, yy, batch_size=None, shuffle=False):
    """
    Creates a PyTorch DataLoader from input arrays and target values.

    Parameters
    ----------
    arrays : list of array-like
        List of input arrays containing the model features.
    yy : array-like
        Target values associated with the input samples.
    batch_size : int, optional
        Number of samples per batch. If None, the batch size is set to
        min(200, len(yy)), following the default behavior used by sklearn.
    shuffle : bool, default=False
        Whether to shuffle the samples at the beginning of each epoch.

    Returns
    -------
    torch.utils.data.DataLoader
    DataLoader containing the input arrays and target values.
    """

    # The batch size is defined in this way so that it is the same as that of sklearn

    if batch_size is None:

        batch_size = min(200, len(yy))

    tensors = [torch.tensor(a, dtype=torch.float32) for a in arrays] + [
        torch.tensor(yy, dtype=torch.float32).unsqueeze(1)
    ]

    dataset = TensorDataset(*tensors)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# =======================================================================================================
# =======================================================================================================


def scale_split(arrays, tr_idx, eval_idx):
    """
    Scales input arrays using statistics fitted exclusively on the training data.

    Parameters
    ----------
    arrays : list of array-like
        List of input arrays to be scaled. The first dimension must correspond
        to the samples.
    tr_idx : array-like
        Indices of the samples used to fit the StandardScaler.
    eval_idx : array-like
        Indices of the samples used to transform the evaluation data.

    Returns
    -------
    scaled_tr : list of numpy.ndarray
        List of scaled training arrays.
    scaled_eval : list of numpy.ndarray
        List of scaled evaluation arrays.

    Notes
    -----
    A separate StandardScaler is fitted for each input array. The scaler is
    fitted only on the training data to avoid data leakage, and the same
    transformation is then applied to the evaluation data.
    """

    scaled_tr = []
    scaled_eval = []

    for X in arrays:

        scaler = StandardScaler()

        # The training data is used to adjust the scaler ...
        X_train_scaled = scaler.fit_transform(X[tr_idx])
        # ... and then that adjustment is applied to the evaluation data
        X_eval_scaled = scaler.transform(X[eval_idx])

        scaled_tr.append(X_train_scaled)
        scaled_eval.append(X_eval_scaled)

    return (scaled_tr, scaled_eval)


# =======================================================================================================
# =======================================================================================================


def train_epoch(model, loader, optimizer, criterion):
    """
    Trains the model for one epoch using mini-batch gradient descent.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model to be trained.
    loader : torch.utils.data.DataLoader
        DataLoader providing the input batches and corresponding target values.
    optimizer : torch.optim.Optimizer
        Optimizer used to update the model parameters based on the computed
        gradients.
    criterion : torch.nn.Module
        Loss function used to measure the difference between the model
        predictions and the target values.

    Returns
    -------
    None
        The model parameters are updated in place during training.


    Notes
    -----
    Gradient clipping is intentionally not applied in order to remain closer
    to the training behavior of sklearn.
    """

    model.train()

    for *Xs, yb in loader:

        optimizer.zero_grad()

        logits = model(*Xs)

        loss = criterion(logits, yb)

        loss.backward()

        # NO GRADIENT CLIPPING: To get closer to the behavior of sklearn

        # torch.nn.utils.clip_grad_norm_(
        #     model.parameters(),
        #     max_norm=1.0
        # )

        optimizer.step()


# =======================================================================================================
# =======================================================================================================


@torch.no_grad()
def predict(model, loader):
    """
    Generates probability predictions and retrieves the corresponding labels.

    Parameters
    ----------
    model : torch.nn.Module
        Trained neural network model used to generate predictions.
    loader : torch.utils.data.DataLoader
        DataLoader providing the input batches and their corresponding labels.

    Returns
    -------
    preds : numpy.ndarray
        Predicted probabilities for each sample, obtained by applying the
        sigmoid function to the model logits.
    labels : numpy.ndarray
        Ground-truth labels corresponding to each sample.
    """

    model.eval()

    preds = []  # model predictions
    labels = []  # real labels

    for *Xs, yb in loader:

        logits = model(*Xs)
        probabilities = torch.sigmoid(logits)

        preds.append(probabilities)
        labels.append(yb)

    preds = torch.cat(preds).cpu().numpy().squeeze()
    labels = torch.cat(labels).cpu().numpy().squeeze().astype(int)

    return (preds, labels)


# =======================================================================================================
# =======================================================================================================


def nested_cv_fusion(
    features_arrays, y, param_grid, outer_splits=5, inner_splits=3, epochs=60
):
    """
    Performs nested stratified cross-validation for a fusion neural network.

    Parameters
    ----------
    features_arrays : list of numpy.ndarray
        List of feature arrays used as inputs to the fusion model. Each array
        must contain the same number of samples in its first dimension.
    y : numpy.ndarray
        Target labels for each sample.
    param_grid : dict
        Dictionary containing the hyperparameters and their candidate values
        to evaluate during the inner cross-validation.
    outer_splits : int, default=5
        Number of folds used in the outer cross-validation.
    inner_splits : int, default=3
        Number of folds used in the inner cross-validation for hyperparameter
        selection.
    epochs : int, default=60
        Number of training epochs for each model.

    Returns
    -------
    outer_auc_scores : list of float
        AUC scores obtained on the outer test set for each outer fold.
    best_params_per_fold : list of dict
        Best hyperparameter configuration selected during the inner
        cross-validation for each outer fold.
    outer_histories : list of dict
        Training history for each outer fold, including train and test AUC
        scores, predictions, and labels for each epoch.
    """

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    feature_dims = [X.shape[1] for X in features_arrays]

    # -------------------------------------------------------------------------------------------------------------------
    # OUTER CV

    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=42)

    outer_auc_scores = []
    best_params_per_fold = []
    outer_histories = []

    print(
        f"Evaluating {len(combinations)} combinations ({outer_splits * len(combinations) * inner_splits} internal training)"
    )

    for outer_fold, (train_outer_idx, test_outer_idx) in enumerate(
        outer_cv.split(features_arrays[0], y)
    ):

        print()
        print("=" * 70)
        print(f"OUTER FOLD " f"{outer_fold + 1}/" f"{outer_splits}")
        print("=" * 70)

        # Remember: features_arrays can be a list of arrays
        arrays_outer_tr = [X[train_outer_idx] for X in features_arrays]
        y_outer_tr = y[train_outer_idx]

        best_inner_auc = -1.0
        best_inner_params = None

        # -------------------------------------------------------------------------------------------------------------------
        # INNER CV

        for params in combinations:

            inner_cv = StratifiedKFold(
                n_splits=inner_splits, shuffle=True, random_state=(42 + outer_fold)
            )

            inner_scores = []

            for tr_inner_idx, val_inner_idx in inner_cv.split(
                arrays_outer_tr[0], y_outer_tr
            ):

                # The feature arrays are scaled, training the scaler exclusively on the train set
                scaled_inner_tr, scaled_inner_val = scale_split(
                    arrays_outer_tr, tr_inner_idx, val_inner_idx
                )

                train_inner_loader = make_loader(
                    scaled_inner_tr, y_outer_tr[tr_inner_idx], shuffle=True
                )
                val_inner_loader = make_loader(
                    scaled_inner_val, y_outer_tr[val_inner_idx], shuffle=False
                )  # Shuffle is not necessary during validation

                # Model is initialized
                model = FusionNet(feature_dims, p_drop=0.0)

                # We used the Adam optimizer to adjust to the sklearn configuration.
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=1e-3,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=(params["weight_decay"]),
                )

                criterion = nn.BCEWithLogitsLoss()

                # Model training
                for _ in range(epochs):
                    train_epoch(model, train_inner_loader, optimizer, criterion)

                # The predictions for the model trained on the inner fold are calculated and the AUC is saved.
                preds, labels = predict(model, val_inner_loader)
                val_auc = roc_auc_score(labels, preds)
                inner_scores.append(val_auc)

            # Mean of the AUC of the inner fold
            mean_inner_auc = np.mean(inner_scores)

            print(
                f"weight_decay = {params['weight_decay']} -> AUC inner = {mean_inner_auc:.4f}"
            )

            # The average AUC value of the inner folds is saved if it is better than the one already saved;
            # in that case, the parameters used are also saved.
            if mean_inner_auc > best_inner_auc:
                best_inner_auc = mean_inner_auc
                best_inner_params = params.copy()

        print()
        print("Better parameters:", best_inner_params)
        print(f"Mean inner AUC: " f"{best_inner_auc:.4f}")

        # External evaluation

        scaled_outer_tr, scaled_outer_test = scale_split(
            features_arrays, train_outer_idx, test_outer_idx
        )

        # Loaders are created for the train and test sets.
        train_outer_loader = make_loader(
            scaled_outer_tr, y[train_outer_idx], shuffle=True
        )
        train_eval_loader = make_loader(
            scaled_outer_tr, y[train_outer_idx], shuffle=False
        )  # This loader is created without shuffle to facilitate performance analysis

        test_outer_loader = make_loader(
            scaled_outer_test, y[test_outer_idx], shuffle=False
        )

        # Final model is initialized
        final_model = FusionNet(feature_dims, p_drop=0.0)

        optimizer = torch.optim.Adam(
            final_model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=(best_inner_params["weight_decay"]),  # best values ​​found
        )

        criterion = nn.BCEWithLogitsLoss()

        history = {
            "train_auc": [],
            "train_pred": [],
            "train_labels": [],
            "test_auc": [],
            "test_pred": [],
            "test_labels": [],
        }

        # Final training
        for _ in range(epochs):

            train_epoch(final_model, train_outer_loader, optimizer, criterion)

            # train
            tr_preds, tr_labels = predict(final_model, train_eval_loader)
            tr_auc = roc_auc_score(tr_labels, tr_preds)

            # test
            te_preds, te_labels = predict(final_model, test_outer_loader)
            te_auc = roc_auc_score(te_labels, te_preds)

            # save results
            history["train_auc"].append(tr_auc)
            history["train_pred"].append(tr_preds.copy())
            history["train_labels"].append(tr_labels.copy())
            history["test_auc"].append(te_auc)
            history["test_pred"].append(te_preds.copy())
            history["test_labels"].append(te_labels.copy())

        # save final results
        final_test_auc = history["test_auc"][-1]

        outer_auc_scores.append(final_test_auc)
        best_params_per_fold.append(best_inner_params)
        outer_histories.append(history)

        print()
        print(f"AUC Outer Test " f"Fold {outer_fold + 1}: " f"{final_test_auc:.4f}")

    # print the final results
    print()
    print("=" * 70)
    print("FINAL RESULTS!!")
    print("=" * 70)
    print(
        f"Mean AUC per folds = {np.mean(outer_auc_scores):.4f} (+/- {np.std(outer_auc_scores):.4f})"
    )
    print("=" * 70)

    return (outer_auc_scores, best_params_per_fold, outer_histories)
