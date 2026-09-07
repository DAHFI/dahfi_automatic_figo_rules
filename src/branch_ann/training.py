from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import itertools
import numpy as np
import torch
import torch.nn as nn

from .model import FusionNet
from .analisis import permutation_importance
from .utils import scale_split, make_loader, train_epoch, predict

# =======================================================================================================
# =======================================================================================================


def nested_cv_fusion(
    features_arrays,
    y,
    param_grid,
    outer_splits=5,
    inner_splits=3,
    epochs=60,
    make_permutation_importance=False,  # Perform an analysis of how the variation in the values ​​of each feature influences the prediction of the model
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

    print("INFO:")
    print("      ->  A permutation importance analysis will be performed")

    # -------------------------------------------------------------------------------------------------------------------

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    feature_dims = [X.shape[1] for X in features_arrays]

    # -------------------------------------------------------------------------------------------------------------------
    # OUTER CV

    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=42)

    outer_auc_scores = []
    best_params_per_fold = []
    outer_histories = []

    # It will be used in the case of a permutation importance analysis.
    permutation_importances_per_fold = []

    print(
        f"      ->  Evaluating {len(combinations)} combinations ({outer_splits * len(combinations) * inner_splits} internal training)"
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

        # We want the permutation importance analysis to be performed
        if make_permutation_importance:
            _, importances = permutation_importance(
                final_model,
                scaled_outer_test,
                y[test_outer_idx],
                n_repeats=20,
            )

            permutation_importances_per_fold.append(importances)

            # branch_names = ["FHR", "UC", "Clinical"]

            # print(f"\nBaseline AUC: {baseline_auc:.4f}")

            # for branch_name, branch_imp in zip(branch_names, importances):
            #     print(f"\n{branch_name}")

            #     for result in branch_imp:
            #         print(
            #             f"Feature {result['feature']:2d}: "
            #             f"{result['mean']:.4f} "
            #             f"+/- {result['std']:.4f}"
            #         )

        # Information on the weight that the outputs of each branch have in the final prediction of the model
        # if final_model.use_branches:
        #     # weights of the first layer of the classifier
        #     W = final_model.classifier[0].weight.detach().cpu()
        #     branch_size = 16

        #     print("\nBranch weights:")
        #     for i in range(len(feature_dims)):
        #         start = i * branch_size
        #         end = (i + 1) * branch_size

        #         W_branch = W[:, start:end]
        #         importance = W_branch.abs().mean().item()

        #         print(f"Branch {i + 1}: {importance:.6f} -> {W_branch}")

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

    return (
        outer_auc_scores,
        best_params_per_fold,
        outer_histories,
        permutation_importances_per_fold,
    )
