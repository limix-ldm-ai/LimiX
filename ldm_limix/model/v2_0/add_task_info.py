import torch
import torch.nn as nn


class AddTaskInfo(nn.Module):
    """Add task-type embeddings to the y tokens."""

    def __init__(
            self,
            *,
            embedding_size: int = 192,
            num_y_tokens: int = 1,
    ):
        super().__init__()
        if num_y_tokens <= 0:
            raise ValueError("num_y_tokens must be greater than 0")
        self.num_y_tokens = num_y_tokens
        self.embedding_size = embedding_size
        # Size 3 is kept for checkpoint compatibility; only 0/1 (cls/reg) are used.
        self.token_type_embedding = nn.Embedding(3, self.embedding_size)

    def forward(self, transformer_input: torch.Tensor, y_type: torch.Tensor) -> torch.Tensor:
        # All trailing y tokens belong to task semantics.
        x = transformer_input[..., :-self.num_y_tokens, :]
        y = transformer_input[..., -self.num_y_tokens:, :]
        y = y + self.token_type_embedding(y_type.long()).unsqueeze(-2)
        return torch.cat((x, y), dim=-2)
