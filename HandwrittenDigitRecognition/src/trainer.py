import torch
from torch.utils.data import DataLoader


class MyTrainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self._check_optim_net_aligned()

    # Ensures that the given optimizer points to the given model
    def _check_optim_net_aligned(self):
        assert self.optimizer.param_groups[0]['params'] == list(self.model.parameters())

    # Trains the model
    def fit(self,
            train_dataloader: DataLoader,
            test_dataloader: DataLoader,
            epochs: int = 50,
            eval_every: int = 1,
            early_stopping: bool = True,
            sub_epoch_logs: bool = False,
            sub_epoch_percentile: float = 0.1):

        # Stores the current best loss, used to abort training early,
        # helps to prevent over-fitting
        best_loss = 1e9

        for e in range(epochs):
            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            print(len(train_dataloader))
            for i, data in enumerate(train_dataloader):
                # Every data instance is an input + label pair
                inputs, labels = data

                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # Zero your gradients for every batch!
                self.optimizer.zero_grad()
                # Make predictions for this batch
                outputs = self.model(inputs)
                # Compute the loss and its gradients
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                # Adjust learning weights
                self.optimizer.step()

            # Reports on the training progress
            if (e + 1) % eval_every == 0:
                torch.save(self.model.state_dict(), "check_points/model_checkpoint_e" + str(e + 1) + ".pt")
                with torch.no_grad():
                    self.model.eval()
                    losses = []
                    for i, data in enumerate(test_dataloader):
                        # Every data instance is an input + label pair
                        inputs, labels = data

                        if torch.cuda.is_available():
                            inputs = inputs.cuda()
                            labels = labels.cuda()

                        output = self.model(inputs)
                        loss = self.loss_fn(output, labels)
                        losses.append(loss.item())

                    avg_loss = torch.Tensor(losses).mean().item()

                    if early_stopping:
                        if avg_loss < best_loss:
                            best_loss = avg_loss
                            print("The loss after", (e + 1), "epochs was", round(avg_loss, 4))
                        else:
                            print("The loss after", (e + 1), "epochs was", round(avg_loss, 4))
                            print("The loss had increased since the last checkpoint, aborting training!")
                            break
                    else:
                        print("The loss after", (e + 1), "epochs was", round(avg_loss, 4))
