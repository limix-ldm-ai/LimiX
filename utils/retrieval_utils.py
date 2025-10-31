import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder


class RelabelRetrievalY:
    def __init__(self, y_train: torch.Tensor):
        self.y_train = y_train.cpu().numpy()
        self.label_encoders = [LabelEncoder() for i in range(y_train.shape[0])]

    def transform_y(self, ):
        for i in range(self.y_train.shape[0]):
            self.y_train[i] = np.expand_dims(self.label_encoders[i].fit_transform(self.y_train[i].ravel()), axis=1)
        self.label_y = self.y_train.copy().astype(np.int32)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32,
                                    device=torch.device('cuda') if torch.cuda.is_available() else 'cpu')
        return self.y_train

    def inverse_transform_y(self, X: np.ndarray) -> np.ndarray:
        for i in range(X.shape[0]):
            batch_label = np.unique(self.label_y[i])
            reverse_perm = self.label_encoders[i].inverse_transform(batch_label).astype(np.int32)
            reverse_output = np.full_like(X[i], fill_value=-np.inf)
            reverse_output[reverse_perm] = X[i, batch_label]
            X[i] = reverse_output
        return X


if __name__ == '__main__':
    y_train = torch.tensor([[[7],[7],[8], [5]],[[4], [3],[3], [6]]])
    output = np.array([[0.2, 2, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                       [0.2, 2, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]],dtype=np.float32)

    relabel = RelabelRetrievalY(y_train)
    y_train, label_y = relabel.transform_y()
    output = relabel.inverse_transform_y(output)
