from base_classes import BaseClassifier
from configs.config import CFG

import torch
from tqdm.auto import tqdm

class TinyVGGClassifier(BaseClassifier):

    def __init__(self, CFG):
        super().__init__()
        self.model=None
        self.train_data=None
        self.test_data=None
        self.results=None

    def load_data(self):
        pass

    def build(self):
        pass

    def train(self, model: torch.nn.Module,
              train_dataloader: torch.utils.data.DataLoader,
              test_dataloader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer,
              loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),  # default for multi-class classifier
              epochs: int = 5,
              device=device):
        # create an empty results dictionary
        results = {"train_loss": [],
                   "train_acc": [],
                   "test_loss": [],
                   "test_acc": []}

        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = train_step(model=model,
                                               dataloader=train_dataloader,
                                               loss_fn=loss_fn,
                                               optimizer=optimizer,
                                               device=device)

            test_loss, test_acc = test_step(model=model,
                                            dataloader=test_dataloader,
                                            loss_fn=loss_fn,
                                            device=device)
            # 4. Print out what's happening
            print(
                f"Epoch: {epoch + 1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"test_loss: {test_loss:.4f} | "
                f"test_acc: {test_acc:.4f}"
            )

            # 5. Update results dictionary
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

        # 6. Return the filled results at the end of the epochs
        return results

    def evaluate(self):
        pass