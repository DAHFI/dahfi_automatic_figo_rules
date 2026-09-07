import torch
import torch.nn as nn


class BranchNet(nn.Module):
    """
    Neural network branch used to process one input feature set.

    The branch transforms the input features through two fully connected
    layers and produces a 16-dimensional embedding that is later used by
    the fusion classifier.

    Parameters
    ----------
    in_dim : int
        Number of input features.
    p_drop : float, default=0.0
        Dropout probability. Currently included for interface consistency
        but not used in the network architecture.

    Notes
    -----
    The architecture is:

        Input -> Linear(64) -> ReLU -> Linear(16) -> ReLU

    The output of the branch is therefore a 16-dimensional representation.
    """

    def __init__(self, in_dim, p_drop=0.0):

        super().__init__()

        # input -> 64 -> ReLU -> 16 -> ReLU -> FusionNet

        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 16), nn.ReLU()
        )

    def forward(self, x):

        return self.net(x)


class FusionNet(nn.Module):
    """
    Neural network that combines multiple feature sources for classification.

    When multiple feature sets are provided, each feature source is processed
    independently by a BranchNet and the resulting embeddings are concatenated
    before being passed to the final classifier. When only one feature source
    is provided, it is passed directly to the classifier.

    Parameters
    ----------
    feature_dims : list of int
        Number of input features for each feature source. Each element
        corresponds to one input branch.
    p_drop : float, default=0.0
        Dropout probability passed to the BranchNet instances. Currently
        included for interface consistency but not used in the network
        architecture.

    Notes
    -----
    For multiple feature sources, the architecture is:

        Feature 1 -> BranchNet --\\
        Feature 2 -> BranchNet ----> Concatenation -> Classifier -> Logit
        ...                       /
        Feature N -> BranchNet --/

    Each BranchNet produces a 16-dimensional embedding. Therefore, the
    classifier receives ``16 * len(feature_dims)`` features.

    For a single feature source, the input is passed directly to the
    classifier.

    The final classifier has the following architecture:

        Input -> Linear(64) -> ReLU -> Linear(16) -> ReLU -> Linear(1)

    The final output is a single logit. A sigmoid should be applied externally
    when probabilities are required.
    """

    def __init__(self, feature_dims, p_drop=0.0):

        super().__init__()

        # -------------------------------------------------------------------------------------------------------------------
        # Different branches are created if features from different sources are introduced (FHR, UC, Corr, ...)

        self.use_branches = len(feature_dims) > 1

        if self.use_branches:

            self.ramas = nn.ModuleList([BranchNet(dim, p_drop) for dim in feature_dims])

            in_clf_dim = 16 * len(
                feature_dims
            )  # 16 puesto que es el tamaño de la capa final de las ramas

        else:

            in_clf_dim = feature_dims[0]

        # -------------------------------------------------------------------------------------------------------------------
        # input -> 64 -> ReLU -> 16 -> ReLU -> 1

        self.classifier = nn.Sequential(
            nn.Linear(in_clf_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, *inputs):

        if self.use_branches:

            # The output from all branches is concatenated to pass it to FusionNet

            embeddings = [branch(x) for branch, x in zip(self.ramas, inputs)]

            x = torch.cat(embeddings, dim=1)

        else:

            x = inputs[0]

        return self.classifier(x)
