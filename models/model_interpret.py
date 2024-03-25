import torch
from torch import nn

class Model_Interpret(nn.Module):
    def __init__(self, n_embed: int, n_layer: int):
        """
        Initializes the interpretation model.

        Args:
            n_embed (int): The number of embeddings in the input.
            n_layer (int): The number of layers in the output.
        """
        super(Model_Interpret, self).__init__()
        self.n_embed = n_embed
        self.n_layer = n_layer
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def save(self, name: str):
        """
        Saves the model to a file.

        Args:
            name (str): The name of the file to save the model to.
        """
        pass

class Model_LogisticRegression(Model_Interpret):
    def __init__(self, n_embed: int, n_layer: int):
        """
        Initializes the logistic regression model.

        Args:
            n_embed (int): The number of embeddings in the input.
            n_layer (int): The number of layers in the output.
        """
        super(Model_LogisticRegression, self).__init__(n_embed, n_layer)
        self.linear = nn.Linear(n_embed, n_layer)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the logistic regression model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.linear(x)

    def save(self, name: str):
        """
        Save self.linear to a file.
        """
        torch.save(self.linear, name)
        print(f"Model saved to {name}")