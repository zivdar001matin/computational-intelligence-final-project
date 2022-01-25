import pytorch_lightning as pl
from argparse import ArgumentParser
import torch
from torch.nn import functional as F

try:
    from torchmetrics.functional import accuracy
except ImportError:
    from pytorch_lightning.metrics.functional import accuracy


class ImageClassificationBase(pl.LightningModule):
    """
    A LightningModule is a torch.nn.Module but with added functionality. Use it as such!
    # define base methods for out later use 
    # it is not defining important methods (init and forward) yet.
    """

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--epochs",
            type=int,
            default=20,
            metavar="E",
            help="number of passes (default: 20)",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=128,
            metavar="N",
            help="input batch size for training (default: 128)",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=4,
            metavar="N",
            help="number of workers (default: 4)",
        )
        parser.add_argument(
            "--lr",
            type=float,
            default=0.01,
            metavar="LR",
            help="learning rate (default: 0.01)",
        )
        parser.add_argument(
            "--validation_ratio",
            type=float,
            default=0.1,
            metavar="N",
            help="validation split ratio (default: 0.1)",
        )
        parser.add_argument(
            "--test_ratio",
            type=float,
            default=0.1,
            metavar="N",
            help="test split ratio (default: 0.1)",
        )
        parser.add_argument(
            "--grad_clip",
            type=float,
            default=0.1,
            metavar="N",
            help="gradient clipping threshold (default: 0.1)",
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=1e-4,
            metavar="N",
            help="weight decay number (default: 1e-4)",
        )
        return parser

    def cross_entropy_loss(self, logits, labels):
        """
        Initializes the loss function
        :return: output - Initialized cross entropy loss function
        """
        return F.nll_loss(logits, labels)
        # return F.cross_entropy(logits, labels)
    
    def training_step(self, train_batch, batch_idx):
        """
        Training the data as batches and returns training loss on each batch
        :param train_batch: Batch data
        :return: output - Training loss
        """
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        return {"loss": loss}
    
    def validation_step(self, val_batch, batch_idx):
        """
        Performs validation of data in batches
        :param val_batch: Batch data
        :return: output - valid step loss
        """
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        acc = accuracy(logits, y)
        return {
            # calculate loss
            'val_step_loss': loss,
            # and accuracy
            'val_step_acc': acc
        }
    
    def validation_epoch_end(self, outputs):
        """
        Computes average validation accuracy
        :param outputs: outputs after every epoch end
        :return: output - average valid loss

        after the end of epoch
        return validation loss and accuracy for whole epoch
        calculated from combining
        """
        batch_losses = [x['val_step_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_step_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        self.log("val_step_loss", epoch_loss.item(), sync_dist=True)
        self.log("val_step_acc", epoch_acc.item(), sync_dist=True)
        return {
            'val_step_loss': epoch_loss.item(),
            'val_step_acc': epoch_acc.item()
        }
    
    def epoch_end(self, epoch, result):
        """
        print data about the current (last) epoch 
        it helps the observer see everything is going fine
        """
        print(
            f"=> #{epoch}, "
            f"train_loss: {result['train_loss']:.3f}, "
            f"val_step_loss: {result['val_step_loss']:.3f}, "
            f"val_step_acc: {result['val_step_acc']:.3f}"
        )
    
    def test_step(self, test_batch, batch_idx):
        """
        Performs test and computes the accuracy of the model
        :param test_batch: Batch data
        :return: output - Testing accuracy
        """
        x, y = test_batch
        output = self.forward(x)
        _, y_hat = torch.max(output, dim=1)
        test_acc = accuracy(y_hat.cpu(), y.cpu())
        return {"test_acc": test_acc}

    def test_epoch_end(self, outputs):
        """
        Computes average test accuracy score
        :param outputs: outputs after every epoch end
        :return: output - average test loss
        """
        avg_test_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        self.log("avg_test_acc", avg_test_acc)