
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorchvideo.models.resnet
import os
import pytorch_lightning
import pytorchvideo.data
import torch.utils.data

def make_kinetics_resnet():
  return pytorchvideo.models.resnet.create_resnet(
    input_channel=1,
    model_depth=50,  # Choix arbitraire, pourrait être modifié selon les besoins.
    model_num_class=2,  # Classification binaire.
    stem_conv_kernel_size=(3, 3, 3),
    stem_conv_stride=(1, 1, 1),
    stem_pool_kernel_size=(1, 2, 2),
    stem_pool_stride=(1, 2, 2),
    head_pool_kernel_size=(4, 2, 2),
)


class VideoClassificationLightningModel(pytorch_lightning.LightningModule):
  def __init__(self):
      super().__init__()
      self.model = make_kinetics_resnet().to("cuda")

  def forward(self, x):
      return self.model(x)

  def training_step(self, batch, batch_idx):
      # The model expects a video tensor of shape (B, C, T, H, W), which is the
      # format provided by the dataset
      y_hat = self.model(batch["videos"])

      # Compute cross entropy loss, loss.backwards will be called behind the scenes
      # by PyTorchLightning after being returned from this method.
      loss = F.cross_entropy(y_hat, batch["label"])

      # Log the train loss to Tensorboard
      self.log("train_loss", loss.item())

      return loss

  def validation_step(self, batch, batch_idx):
    print(type(batch))  # Vérifier le type de 'batch'
    if isinstance(batch, dict):
        print(batch.keys())  # Si c'est un dictionnaire, afficher ses clés
    y_hat = self.model(batch["video"])
    y_hat = self.model(batch["videos"])
    loss = F.cross_entropy(y_hat, batch["label"])
    self.log("val_loss", loss)
    return loss

  def configure_optimizers(self):
      """
      Setup the Adam optimizer. Note, that this function also can return a lr scheduler, which is
      usually useful for training video models.
      """
      return torch.optim.Adam(self.parameters(), lr=1e-1)
