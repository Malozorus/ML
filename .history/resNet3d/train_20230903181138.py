from datamodule import VideoDataModule, video_resize_and_sample
from model import VideoClassificationLightningModel
import pytorch_lightning
import warnings
import torch





if __name__ == "__main__":

    torch.set_float32_matmul_precision('medium')
    transform = video_resize_and_sample
    classification_module = VideoClassificationLightningModel()
    video_data_module = VideoDataModule(transform=transform)
    video_data_module.setup()
    trainer = pytorch_lightning.Trainer(max_epochs=10, log_every_n_steps=1)
    trainer.fit(classification_module, video_data_module)
    trainer.test(classification_module, video_data_module.val_dataloader())
    classification_module.save_pretrained("C:/Users/mcouv/Work/machine-learning/ML_tuto/pretrained_models/model.pth")