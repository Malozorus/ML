from datamodule import VideoDataModule, video_resize_and_sample
from model import VideoClassificationLightningModel
import pytorch_lightning
import warnings
import torch



def train():
    # Initialisation du VideoDataModule avec la transformation désirée
    
    transform = video_resize_and_sample
    video_data_module = VideoDataModule(transform=transform)
    video_data_module.setup()
    classification_module = VideoClassificationLightningModel()
    video_data_module = VideoDataModule(transform=transform)
    video_data_module.setup()
    trainer = pytorch_lightning.Trainer(max_epochs=10)
    trainer.fit(classification_module, video_data_module)

if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=UserWarning, module='torchaudio.backend.utils')
    torch.set_float32_matmul_precision('medium')
    train()