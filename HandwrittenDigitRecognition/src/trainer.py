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
        # Used to perform sub batch eval
        batch_log_every = len(train_dataloader) * sub_epoch_percentile

        for e in range(epochs):
            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)

            # Stores data about the batch
            batch_losses = []
            sub_batch_losses = []

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

                # Saves data
                batch_losses.append(loss)
                sub_batch_losses.append(loss)

                # Performs sub batch log
                if sub_epoch_logs and (i + 1) % batch_log_every == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
                        e, (i + 1) * train_dataloader.batch_size,
                        len(train_dataloader) * train_dataloader.batch_size,
                        100.0 * (i + 1) / len(train_dataloader), torch.Tensor(sub_batch_losses).mean()))
                    sub_batch_losses.clear()

            # Reports on the path
            print('Train Epoch: {} Average Loss: {:.6f}'.format(e, torch.Tensor(batch_losses).mean()))

            # Reports on the training progress
            if (e + 1) % eval_every == 0:
                torch.save(self.model.state_dict(), "model_checkpoint_e" + str(e + 1) + ".pt")
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
                            print("\nThe loss after", (e + 1), "epochs was", round(avg_loss, 4), "\n")
                        else:
                            print("\nThe loss after", (e + 1), "epochs was", round(avg_loss, 4))
                            print("The loss had increased since the last checkpoint, aborting training!", "\n")
                            # revers to the previse model sice the current one is worse
                            self.model.load_state_dict(torch.load("check_points/model_checkpoint_e" + str(e) + ".pt"))
                            break
                    else:
                        print("\nThe loss after", (e + 1), "epochs was", round(avg_loss, 4), "\n")
