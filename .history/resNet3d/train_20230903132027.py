from datamodule import VideoDataModule, video_resize_and_sample, VideoClassificationLightningModule
import model
import pytorch_lightning

def train():
    # Initialisation du VideoDataModule avec la transformation désirée
    transform = video_resize_and_sample
    video_data_module = VideoDataModule(transform=transform)
    video_data_module.setup()
    classification_module = VideoClassificationLightningModule()
    video_data_module = VideoDataModule(transform=transform)
    video_data_module.setup()
    trainer = pytorch_lightning.Trainer()
    trainer.fit(classification_module, video_data_module)

if __name__ == "__main__":
    train()