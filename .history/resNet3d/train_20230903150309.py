from datamodule import VideoDataModule, video_resize_and_sample
from model import VideoClassificationLightningModel, SimpleVideoNet
import pytorch_lightning
import warnings
import torch



def train():
    # Initialisation du VideoDataModule avec la transformation désirée
    
    transform = video_resize_and_sample
    video_data_module = VideoDataModule(transform=transform)
    video_data_module.setup()
    classification_module = SimpleVideoNet()
    video_data_module = VideoDataModule(transform=transform)
    video_data_module.setup()
    trainer = pytorch_lightning.Trainer(max_epochs=10, log_every_n_steps=1)
    trainer.fit(classification_module, video_data_module)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    train()