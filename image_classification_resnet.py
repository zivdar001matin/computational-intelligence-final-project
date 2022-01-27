from image_classification_base import ImageClassificationBase
from cars_data_module import CarsDataModule

import mlflow.pytorch
from mlflow.models.signature import infer_signature
from PIL import Image

import os
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

import torch
import torch.nn as nn

def conv_block(in_channels, out_channels, pool=False, stride=1, padding=1):
    """
    apply conv2s + batch normalization and relu (and max pool if needed)
    """
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride ,padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    # compile operations
    return nn.Sequential(*layers)

class ResNet(ImageClassificationBase):
    def __init__(self, num_classes, **kwargs):
        """
        constructor: create layers of cnn
        cleaning code with conv_block function
        """
        super().__init__()

        self.optimizer = None
        self.scheduler = None
        # images are (3, 64, 64) (channels, width, height)
        self.conv1 = conv_block(3, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(
            conv_block(128, 128),
            conv_block(128, 128)
        )
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(
            conv_block(512, 512),
            conv_block(512, 512)
        )
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            # flatten the model
            nn.Flatten(),
            # linear transformation : classify  num of classes
            nn.Linear(2048, num_classes)
        )
        self.args = kwargs

    def forward(self, xb):
        """
        :param xb: Input data
        :return: output - car label for the input image

        we do not call this method directly 
        this is being called implicitly
        feed input to layers and calculate outputs of each step 
        finally classify the output 
        """
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out

        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out

        return self.classifier(out)
    
    def configure_optimizers(self):
        """
        Initializes the optimizer and learning rate scheduler
        :return: output - Initialized optimizer and scheduler
        """
        # Set up cutom optimizer with weight decay
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.args["learning_rate"], weight_decay=self.args["weight_decay"])

        # one cycle learning rate scheduler
        # update learning rate batch by batch
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.2,
                patience=2,
                min_lr=1e-6,
                verbose=True,
            ),
            "monitor": "val_step_loss",
        }
        return [self.optimizer], [self.scheduler]
    
    # reduce memory use with disabling gradian calculation
    # only if we won't call tensor.backward
    @torch.no_grad()
    def evaluate(self, validation_set_loader):
        """
        evalute model 
        with validation set
        and call validation_step
        """
        self.eval()
        outputs = [self.validation_step(batch) for batch in validation_set_loader]
        return self.validation_epoch_end(outputs)
    
    def fit(self, dm):
        """
        fit the model
        with train set and validation set

        """

        torch.cuda.empty_cache()

        history = []

        for epoch in range(self.args["epochs"]):
            # actually train the mode in this epochs
            self.train()
            train_losses = []
            for batch in dm.train_dataloader():
                loss = self.training_step(batch)
                train_losses.append(loss)
                loss.backward()

                # gradient clipping
                nn.utils.clip_grad_value_(self.parameters(), self.args["grad_clip"])

                self.optimizer.step()
                self.optimizer.zero_grad()

                # update learning rate in optimizer
                self.scheduler.step()

            # validation phase
            result = self.evaluate(self, dm.val_dataloader())
            result['train_loss'] = torch.stack(train_losses).mean().item()
            self.epoch_end(epoch, result)
            history.append(result)
        return history

    def predict(self, df):
        # Convert np.array to PIL
        df['images'] = df['images'].apply(lambda x: Image.fromarray(np.uint8(x)))
        # Tranform images to give to model
        forward_transforms = CarsDataModule.get_transform()
        df['images'] = df['images'].apply(lambda x: forward_transforms(x))
        # Stack dataframe images into one tensor
        xb = torch.stack([df['images'][x] for x in range(len(df['images']))])
        # Call forward function to predict using model
        logits = self.forward(xb)
        predicted_df = pd.DataFrame({
            'predict': logits.argmax(dim=1).numpy()
        })
        predicted_names = predicted_df['predict'].apply(lambda x: CarsDataModule.number_to_label(x))
        return predicted_names

if __name__ == "__main__":
    parser = ArgumentParser(description="PyTorch Autolog Persian Cars Classifier")

    parser = pl.Trainer.add_argparse_args(parent_parser=parser)
    parser = ResNet.add_model_specific_args(parent_parser=parser)

    mlflow.set_tracking_uri("sqlite:///database/mlruns.db")
    mlflow.pytorch.autolog()

    args = parser.parse_args()
    dict_args = vars(args)

    if "accelerator" in dict_args:
        if dict_args["accelerator"] == "None":
            dict_args["accelerator"] = None

    model = ResNet(6, **dict_args)

    dm = CarsDataModule(**dict_args)
    dm.setup(stage=None)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.getcwd(), save_top_k=1, verbose=True, monitor="val_step_loss", mode="min"
    )
    lr_logger = LearningRateMonitor()

    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[lr_logger, checkpoint_callback], checkpoint_callback=True
    )
    trainer.fit(model, dm)
    trainer.test(datamodule=dm)
