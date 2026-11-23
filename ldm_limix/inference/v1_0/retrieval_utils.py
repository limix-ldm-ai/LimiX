import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder


class RelabelRetrievalY:
    def __init__(self, y_train: torch.Tensor):
        """
        Args:
            y_train: (batch_size, n_samples, 1)
        """
        self.y_train = y_train.cpu().numpy()
        self.label_encoders = [LabelEncoder() for i in range(y_train.shape[0])]

    def transform_y(self, ):
        for i in range(self.y_train.shape[0]):
            self.y_train[i] = np.expand_dims(self.label_encoders[i].fit_transform(self.y_train[i].ravel()), axis=1)
        self.label_y = self.y_train.copy().astype(np.int32)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32)
        return self.y_train

    def inverse_transform_y(self, X: np.ndarray, num_classes: int = None) -> np.ndarray:
        """
        Args:
            X: (batch_size,  n_classes)
        Returns:
            X: (batch_size, n_classes)
        """
        if num_classes is None:
            num_classes = 10
        result = np.full((X.shape[0], num_classes), fill_value=-np.inf)
        for i in range(X.shape[0]):
            batch_label = np.unique(self.label_y[i])
            reverse_perm = self.label_encoders[i].inverse_transform(batch_label).astype(np.int32)
            reverse_output = np.full(num_classes, fill_value=-np.inf)
            reverse_output[reverse_perm] = X[i, batch_label]
            result[i] = reverse_output
        return result

def find_top_K_indice(sample_attention: torch.Tensor | np.ndarray, threshold: float = 0.5, mixed_method: str = "max",
                      retrieval_len: int = 200, device: str | torch.device = "cuda"):
    """
        Finds the indices of the largest elements in each row such that their sum
        is at least `threshold` of the total sum of that row.

        Args:
            tensor (torch.Tensor): A 2D tensor with values between 0 and 1.

        Returns:
            list[torch.Tensor[int]]: A list where each element is a list of column indices
                             for the corresponding row in the input tensor.
        """
    num_test_sample, num_train_sample = sample_attention.shape
    result_indices = []

    # 1. Calculate the target sum (90%) for each row
    row_sums = torch.sum(sample_attention, dim=1, keepdim=True)  # Shape: [num_rows, 1]
    target_sums = row_sums * threshold  # Shape: [num_rows, 1]

    for i in range(num_test_sample):
        # Get the current row
        current_row = sample_attention[i]  # Shape: [num_cols]

        # 2. Sort the row in descending order and get the indices
        # torch.sort returns (sorted_values, sorted_indices)
        sorted_values, sorted_indices = torch.sort(current_row, descending=True)

        # 3. Calculate cumulative sum of sorted values
        cumsum_values = torch.cumsum(sorted_values, dim=0)  # Shape: [num_cols]

        # 4. Find the first index where cumulative sum meets or exceeds the target
        # This finds indices where the condition is True, then we take the first one.
        # If the target sum is 0 (row of all zeros), this will select the first element (index 0).
        target_sum_for_row = target_sums[i].item()
        if target_sum_for_row > 0:
            # Use >= for robustness if cumsum exactly equals target
            sufficient_indices_mask = cumsum_values >= target_sum_for_row
            # .nonzero() returns indices of True elements. [0] gets the first one.
            # If no element meets the condition (unlikely with 90%), it might error.
            # We assume at least one element will eventually meet it due to sum >= target.
            try:
                num_elements_needed = torch.nonzero(sufficient_indices_mask, as_tuple=True)[0][0].item() + 1
                if mixed_method == "max":
                    num_elements_needed = min(max(num_elements_needed, retrieval_len), 2000)#max retrieval sample is 2000
                else:
                    num_elements_needed = min(num_elements_needed, retrieval_len)
            except IndexError:
                # Fallback: if no element meets the threshold (e.g., due to floating point),
                # select all elements.
                print(f"Warning: Row {i} might not reach 90% sum with individual elements. Selecting all.")
                num_elements_needed = num_train_sample
        else:
            # If target sum is 0, technically any set including the largest element works.
            # We select the largest element.
            num_elements_needed = 1

        # 5. Get the original column indices of the selected elements
        selected_original_indices = sorted_indices[:num_elements_needed].tolist()

        result_indices.append(torch.tensor(selected_original_indices, device=device))

    return result_indices


def find_top_K_class(X: torch.Tensor, num_class: int = 10):
    unique_classes, counts = torch.unique(X, return_counts=True)

    sorted_indices = torch.argsort(counts, descending=True)
    top_K_classes = unique_classes[sorted_indices[:num_class]]

    all_occurrence_indices = []
    for cls in top_K_classes:
        indices = (X == cls).nonzero(as_tuple=True)[0]
        all_occurrence_indices.append(indices)

    all_indices_flat = torch.cat(all_occurrence_indices) if all_occurrence_indices else torch.tensor([])
    return all_indices_flat


if __name__ == '__main__':
    y_train = torch.tensor([[[7],[7],[8], [5]],[[4], [3],[3], [6]]])
    output = np.array([[0.2, 2, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                       [0.2, 2, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]],dtype=np.float32)

    relabel = RelabelRetrievalY(y_train)
    y_train, label_y = relabel.transform_y()
    output = relabel.inverse_transform_y(output)
