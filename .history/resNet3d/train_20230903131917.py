from datamodule import VideoDataModule, transform, VideoClassificationLightningModule
import model
import pytorch_lightning

def train():
    classification_module = VideoClassificationLightningModule()
    video_data_module = VideoDataModule(transform=transform)
    video_data_module.setup()
    trainer = pytorch_lightning.Trainer()
    trainer.fit(classification_module, video_data_module)

if __name__ == "__main__":
    train()