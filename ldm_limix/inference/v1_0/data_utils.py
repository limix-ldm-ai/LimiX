import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import List, Literal


class TabularInferenceDataset(Dataset):
    """
        A PyTorch Dataset for tabular data inference scenarios.

        This dataset is designed to provide data for inference tasks where
        you might have a fixed training set and varying test samples, optionally
        selecting the training set based on relevance (retrieval) for each test sample.
        When retrieval is used, each test sample (or step) is paired with a specific,
        potentially unique, subset of the training data. When retrieval is not used,
        it's assumed a single, fixed training set is used for all test samples.
        """

    def __init__(self,
                 X_train: torch.Tensor,
                 y_train: torch.Tensor,
                 X_test: torch.Tensor,
                 attention_score: np.ndarray | torch.Tensor = None,
                 retrieval_len: int = 2000,
                 use_retrieval: bool = True,
                 use_cluster: bool = False,
                 cluster_num: int = None,
                 use_threshold: bool = False,
                 mixed_method: str = "max",
                 threshold: float = None
                 ):
        """
                Initializes the TabularInferenceDataset.

                Args:
                    X_train (torch.Tensor): The full set of input training features.
                                            Shape: (num_train_samples, ...).
                    y_train (torch.Tensor): The full set of corresponding training labels.
                                            Shape: (num_train_samples, ...).
                    X_test (torch.Tensor): The set of input features for inference/test samples.
                                           Shape: (num_test_samples, ...).
                    attention_score (np.ndarray, optional): Pre-computed attention scores
                                                            for retrieval logic. Shape depends
                                                            on implementation, e.g., Shape: (num_samples_in_X_train,num_samples_in_X_test).
                                                            Required if use_retrieval is True.
                    retrieval_len (int, optional): The number of top training samples to select
                                                   based on attention scores for each test sample.
                                                   Used only if use_retrieval is True.
                    use_retrieval (bool, optional): Flag to determine data preparation strategy.
                                                    If True, uses attention scores to select relevant training data
                                                    for each test sample.
                                                    If False, assumes a fixed training set is used for all.
                    use_cluster (bool, optional): Flag to determine data preparation strategy.
                                                    If True, uses cluster to select relevant training data
                                                    for each test sample.
                    cluster_num (int, optional): The number of clusters to use.
                """
        self.init_dataset(X_train, y_train, X_test, attention_score, retrieval_len, use_retrieval, use_cluster,
                          cluster_num, use_threshold,mixed_method, threshold)
        # The number of inference steps equals the number of test samples
        self.max_steps = self.X_test.shape[0]
        self.use_retrieval = use_retrieval

    def __len__(self):
        """
                Returns the number of steps/items in the dataset.
                Returns:
                    int: The number of steps, which corresponds to the size of the first dimension
                         of the generated X_test tensor.
                """
        return self.max_steps

    def __getitem__(self, idx: int) -> dict[str, list]:
        """
                Retrieves a single item (data for one inference step) by index.

                Args:
                    idx (int): The index of the test sample/step to retrieve.

                Returns:
                    dict[str, torch.Tensor]: A dictionary containing the data needed for this inference step.
                                             If `use_retrieval` is True, it includes the specific
                                             `X_train`, `y_train`, and `X_test` for this step.
                                             If `use_retrieval` is False, it only includes `X_test`,
                                             as a fixed training set is assumed.
                """
        if self.use_retrieval:
            # Return the specific training data selected for this test sample
            return dict(
                idx=int(idx),
                X_train=self.X_train[idx],  # Training features for this step (retrieved)
                X_test=self.X_test[idx],  # Training labels for this step (retrieved)
                y_train=self.y_train[idx],  # The test sample features
            )
        else:
            # Return only the test data; training data is assumed to be fixed and
            # provided.
            return dict(
                idx=int(idx),
                X_test=self.X_test[idx],
            )

    def init_dataset(self,
                     X_train: torch.Tensor,
                     y_train: torch.Tensor,
                     X_test: torch.Tensor,
                     attention_score:torch.Tensor = None,
                     train_len: int = 2000,
                     use_retrieval: bool = False,
                     use_cluster: bool = False,
                     cluster_num: int = None,
                     use_threshold: bool = False,
                     mixed_method: str = "max",
                     threshold: float = None
                     ):
        #TODO jianshengli Confirm the dimensions of each tensor
        if use_retrieval:
            if use_cluster:
                if use_threshold:
                    top_k_indices = find_top_K_indice(attention_score, threshold=threshold, mixed_method=mixed_method,retrieval_len=train_len)
                    cluster_num = min(cluster_num, len(top_k_indices))
                    cluster_train_sample_indices, cluster_test_sample_indices = cluster_test_data(top_k_indices,
                                                                                                  cluster_num)
                    self.X_train = torch.cat([X_train[x_iter].unsqueeze(0) for x_iter in cluster_train_sample_indices.values()],dim=0)
                    self.y_train = torch.cat([y_train[x_iter].unsqueeze(0) for x_iter in cluster_train_sample_indices.values()],dim=0)
                    self.X_test = torch.cat([X_test[x_iter].unsqueeze(0) for x_iter in cluster_test_sample_indices.values()],dim=0)
                else:
                    top_k_indices = torch.argsort(attention_score)[:, -min(train_len, X_train.shape[0]):]
                    cluster_num = min(cluster_num, len(top_k_indices))
                    cluster_train_sample_indices, cluster_test_sample_indices = cluster_test_data(top_k_indices,
                                                                                                  cluster_num)
                    self.X_train = torch.cat([X_train[x_iter].unsqueeze(0) for x_iter in cluster_train_sample_indices.values()],dim=0)
                    self.y_train = torch.cat([y_train[x_iter].unsqueeze(0) for x_iter in cluster_train_sample_indices.values()],dim=0)
                    self.X_test = torch.cat([X_test[x_iter].unsqueeze(0) for x_iter in cluster_test_sample_indices.values()],dim=0)
            else:
                if use_threshold:
                    top_k_indices = find_top_K_indice(attention_score, threshold=threshold, mixed_method=mixed_method,retrieval_len=train_len)
                else:
                    top_k_indices = torch.argsort(attention_score)[:, -min(train_len, X_train.shape[0]):]
                self.X_train = torch.cat([X_train[x_iter].unsqueeze(0) for x_iter in top_k_indices], dim=0)
                self.y_train = torch.cat([y_train[y_iter].unsqueeze(0) for y_iter in top_k_indices], dim=0).unsqueeze(-1)
                self.X_test = X_test

        else:
            self.X_test = X_test



def cluster_test_data(top_k_indices: torch.Tensor | List[torch.Tensor], k_groups, cluster_method="overlap"):
    """
        Clusters test samples based on their top-k training data indices.
        Handles variable-length 1D tensors in a list.

        Args:
            top_k_indices (torch.Tensor or List[torch.Tensor]):
                - If Tensor: shape (n_test_samples, k) - fixed k.
                - If List[Tensor]: List of 1D tensors, shape (k_i,). k_i can vary.
                                   List length is n_test_samples.
            k_groups (int): The number of clusters.
            cluster_method (str): "overlap" for unique union, else raw indices.

        Returns:
            Tuple[clusters_unions, clusters_sample_indices]:
            - clusters_unions (dict): {cluster_id: union_indices_tensor}
            - clusters_sample_indices (dict): {cluster_id: indices_tensor}. These indices
                                              refer to the position in the original
                                              list/tensor of test samples.
        """
    is_list_input = False

    if isinstance(top_k_indices, list):
        is_list_input = True
        if not top_k_indices:
            raise ValueError("Input list 'top_k_indices' is empty.")
        if not all(isinstance(t, torch.Tensor) and t.dim() == 1 for t in top_k_indices):
            raise TypeError("All elements in the list 'top_k_indices' must be 1D torch.Tensors.")
        n_test_samples = len(top_k_indices)
        # max_train_len is the maximum length of the 1D tensors
        max_train_len = max(t.shape[0] for t in top_k_indices) if n_test_samples > 0 else 0
        # We don't need to create a unified 2D tensor; we process the list directly.
        # processed_top_k_indices conceptually remains the list itself for this path.

    else:  # It's a single 2D tensor
        if not isinstance(top_k_indices, torch.Tensor) or top_k_indices.dim() != 2:
            raise TypeError("Input 'top_k_indices' must be a 2D torch.Tensor or a List of 1D torch.Tensors.")
        top_k_indices = top_k_indices.to("cuda")
        n_test_samples = top_k_indices.shape[0]
        max_train_len = top_k_indices.shape[1]

    if is_list_input:
        # Flatten the list of tensors and find unique indices
        flattened_indices = torch.cat(top_k_indices, dim=0)
    else:
        flattened_indices = top_k_indices.flatten()

    unique_indices = torch.unique(flattened_indices)
    num_unique = len(unique_indices)

    index_to_col = {idx.item(): i for i, idx in enumerate(unique_indices)}

    # Build sparse binary matrix representation
    max_idx = max(index_to_col.keys()) if index_to_col else 0
    mapping_array = torch.zeros(max_idx + 1, dtype=torch.long)
    for k, v in index_to_col.items():
        mapping_array[k] = v

    if is_list_input:
        all_rows_list = []
        all_cols_list = []

        for i, sample_indices_tensor in enumerate(top_k_indices):
            k_i = sample_indices_tensor.shape[0]
            if k_i > 0:

                if sample_indices_tensor.is_cuda:
                    sample_indices_tensor = sample_indices_tensor.cpu()

                mapped_cols = mapping_array[sample_indices_tensor.long()]
                all_rows_list.append(torch.full((k_i,), i, dtype=torch.float))
                all_cols_list.append(mapped_cols)

        if all_rows_list:
            all_rows = torch.cat(all_rows_list)
            all_cols = torch.cat(all_cols_list)
            data = torch.ones(len(all_rows), dtype=torch.float,device="cuda")

            indices = torch.stack([all_rows, all_cols]).to("cuda")
            binary_matrix_sparse = torch.sparse_coo_tensor(
                indices, data,
                size=(n_test_samples, len(index_to_col))
            ).coalesce()


        else:
            indices = torch.empty((2, 0), dtype=torch.float)
            data = torch.empty(0, dtype=torch.float)
            binary_matrix_sparse= torch.sparse_coo_tensor(
                indices, data,
                size=(n_test_samples, len(index_to_col))
            )
    else:
        if top_k_indices.is_cuda:
            indices_tensor = top_k_indices.cpu()
        else:
            indices_tensor = top_k_indices

        n_test_samples, k_fixed = indices_tensor.shape

        flat_indices = indices_tensor.flatten()
        cols = mapping_array[flat_indices.long()]

        rows = torch.repeat_interleave(
            torch.arange(n_test_samples, dtype=torch.float),
            k_fixed
        )

        data = torch.ones(n_test_samples * k_fixed, dtype=torch.float,device="cuda")

        indices = torch.stack([rows, cols]).to("cuda")
        binary_matrix_sparse = torch.sparse_coo_tensor(
            indices, data,
            size=(n_test_samples, len(index_to_col))
        ).coalesce()

    # Create sparse tensor (assuming GPU usage as per original code)
    # Check if we have any data to create the tensor


    # --- Compute overlap matrix ---
    binary_dense = binary_matrix_sparse.to_dense()
    # Matrix multiplication to get pairwise overlaps (n_test_samples x n_test_samples)
    # This works even if the original "k" was variable, as the dense matrix captures the relationship.
    overlap_matrix = torch.mm(binary_dense, binary_dense.t()).to("cuda")

    # Adjust diagonal: overlap of a sample with itself
    # For variable k, the self-overlap is the number of indices for that specific sample.
    diag_indices = torch.arange(n_test_samples, device="cuda")
    if is_list_input:
        # Diagonal entry (i,i) should be the count of indices for sample i
        diag_values = torch.tensor([len(t) for t in top_k_indices], dtype=torch.float, device="cuda")
    else:
        # For fixed k tensor, diagonal is simply k
        diag_values = torch.full((n_test_samples,), max_train_len, dtype=torch.float, device="cuda")

    overlap_matrix[diag_indices, diag_indices] = diag_values

    # --- Perform clustering ---
    # Assuming gpu_kmeans takes (data_points, k) and returns labels
    cluster_labels = gpu_kmeans(binary_dense, k_groups)

    # --- Prepare results ---
    clusters_sample_indices = {}
    clusters_unions = {}

    for cluster_id in range(k_groups):
        # Find sample indices belonging to this cluster
        indices_in_cluster = torch.where(cluster_labels == cluster_id)[0]

        if len(indices_in_cluster) > 0:
            if cluster_method == "overlap":
                # Take the unique union of training indices for samples in this cluster
                if is_list_input:
                    # Concatenate indices from all samples in the cluster, then take unique
                    indices_to_concat = [top_k_indices[i] for i in indices_in_cluster]
                    if indices_to_concat:  # Check if list is not empty
                        union_indices = torch.unique(torch.cat(indices_to_concat, dim=0))
                    else:
                        union_indices = torch.empty(0, dtype=torch.long, device="cuda")  # Or appropriate device
                else:
                    union_indices = torch.unique(top_k_indices[indices_in_cluster, :])

            else:  # cluster_method != "overlap"
                # Take all training indices (may contain duplicates)
                if is_list_input:
                    # Concatenate indices from all samples in the cluster
                    indices_to_concat = [top_k_indices[i] for i in indices_in_cluster]
                    if indices_to_concat:
                        union_indices = torch.cat(indices_to_concat, dim=0)
                    else:
                        union_indices = torch.empty(0, dtype=torch.long, device="cuda")
                else:
                    union_indices = top_k_indices[indices_in_cluster, :].flatten()

            clusters_unions[cluster_id] = union_indices.cpu()
            clusters_sample_indices[cluster_id] = indices_in_cluster.cpu()

    return clusters_unions, clusters_sample_indices


def gpu_kmeans(data, k, max_iters=100, tol=1e-4):
    """
        Kmeans method on gpu
    Args:
        data (torch.Tensor): The input data.
        k (int): The number of clusters.
        max_iters (int, optional): The maximum number of iterations. Defaults to 100.
        tol (float, optional): The tolerance for convergence. Defaults to 1e-4.
    Returns:
        torch.Tensor: The cluster labels.
    """
    n_samples = data.shape[0]
    # init centroids
    data_dense = data.to_dense() if data.is_sparse else data
    centroids = data_dense[torch.randperm(n_samples)[:k]]

    for _ in range(max_iters):
        # calculate distance
        distances = torch.cdist(data_dense, centroids)

        labels = torch.argmin(distances, dim=1)

        new_centroids = torch.stack([
            data_dense[labels == i].mean(dim=0) for i in range(k)
        ])

        if torch.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids

    return labels

def fix_data_shape(X:torch.Tensor,data_type:Literal["feature","label"]="feature",batch_size:int=1):
    assert torch.is_tensor(X) or X is None, f"X should be torch.Tensor, but got {X.type()}"
    if batch_size != 1:
        print(f"fix data with batch_size={batch_size}, please confirm the data shape is (batch_size, seq_len, feature_dim) (feature) or (batch_size, seq_len) (label)")
        return X
    # fix data shape to (batch_size, seq_len, feature_dim) or (batch_size, seq_len)

    if data_type == "feature":
        if X.dim() == 2:
            return X.unsqueeze(0)
        elif X.dim() == 3:
            return X
        else:
            raise ValueError(f"feature should be 2D or 3D tensor with shape (batch_size, seq_len, feature_dim) or (seq_len, feature_dim), but got {X.dim()}D tensor")
    elif data_type == "label":
        X=X.squeeze()
        if X.dim() == 1:
            return X.unsqueeze(0)
        else:
            raise ValueError(f"label should be tensor with shape (batch_size, seq_len, ...) or (seq_len, ...), but got feature_dim>1")




def load_data(data_root,folder):
    le = LabelEncoder()
    train_path = os.path.join(data_root,folder, folder + '_train.csv')
    test_path = os.path.join(data_root,folder, folder + '_test.csv')
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path)
        if os.path.exists(test_path):
            test_df = pd.read_csv(test_path)
        else:
            train_df, test_df = train_test_split(train_df, test_size=0.5, random_state=42)
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            try:
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col])
                X_test[col] = le.transform(X_test[col])
            except Exception as e:
                X_train = X_train.drop(columns=[col])
                X_test = X_test.drop(columns=[col])
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    trainX, trainy = X_train, y_train
    trainX = np.asarray(trainX, dtype=np.float32)
    trainy = np.asarray(trainy, dtype=np.int64)


    testX, testy = X_test, y_test
    testX = np.asarray(testX, dtype=np.float32)
    testy = np.asarray(testy, dtype=np.int64)
    return trainX, trainy, testX, testy
if __name__ == '__main__':
    pass

