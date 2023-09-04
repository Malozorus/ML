from datamodule import VideoDataModule, video_resize_and_sample
from model import VideoClassificationLightningModel
import pytorch_lightning
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='torchaudio.backend.utils')

def train():
    # Initialisation du VideoDataModule avec la transformation désirée
    transform = video_resize_and_sample
    video_data_module = VideoDataModule(transform=transform)
    video_data_module.setup()
    classification_module = VideoClassificationLightningModel()
    video_data_module = VideoDataModule(transform=transform)
    video_data_module.setup()
    trainer = pytorch_lightning.Trainer()
    trainer.fit(classification_module, video_data_module)

if __name__ == "__main__":
    train()