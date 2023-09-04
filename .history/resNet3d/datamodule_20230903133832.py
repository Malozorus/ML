import os
import cv2
import torch
import pytorch_lightning
from torch.utils.data import Dataset
from torchvision.io import read_video
from torch.utils.data import random_split
import random
from torch.utils.data import Subset
from torchvision import transforms


def load_video_with_opencv(video_path):
    vid = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        # Convertir en niveaux de gris (même si c'est déjà le cas)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(torch.tensor(gray_frame))
    vid.release()
    video_tensor = torch.stack(frames)
    return video_tensor.unsqueeze(1)  # Ajoute un canal


class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
        - root_dir (string): Dossier avec toutes les vidéos.
        - transform (callable, optional): Transformation optionnelle à appliquer
            sur une vidéo.
        """
        self.root_dir = root_dir
        self.transform = transform

        self.classes = os.listdir(root_dir)
        self.classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.video_list = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for video_name in os.listdir(cls_dir):
                self.video_list.append((os.path.join(cls_dir, video_name), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_path, label = self.video_list[idx]
        
        # Charger uniquement le flux vidéo (et ignorer l'audio et les timestamps)
        video = load_video_with_opencv(video_path)

        # S'assurer que la vidéo est sur le GPU si disponible
        if torch.cuda.is_available():
            video = video.to("cuda")
        
        if self.transform:
            video = self.transform(video)

        return video, label
    

class VideoDataModule(pytorch_lightning.LightningDataModule):

    # Configuration du Dataset
    _DATA_PATH = "C:/Users/mcouv/Work/machine-learning/ML_tuto/classes"
    _BATCH_SIZE = 8
    _NUM_WORKERS = 16  # Nombre de processus parallèles récupérant les données

    def __init__(self, transform=None):
        super(VideoDataModule, self).__init__()
        self.transform = transform

    def setup(self, stage=None):
        # Initialiser le dataset avec la transformation
        video_dataset = VideoDataset(root_dir=self._DATA_PATH, transform=self.transform)

        # La logique de séparation train/test est déjà définie dans le code donné, donc on la réutilise ici.
        train_indices = []
        test_indices = []

        # Pour chaque classe, récupérez les indices de ses vidéos.
        for cls_name in video_dataset.classes:
            cls_indices = [i for i, (path, label) in enumerate(video_dataset.video_list) if label == video_dataset.class_to_idx[cls_name]]
            
            # Mélangez ces indices.
            random.shuffle(cls_indices)
            
            # Séparez-les en fonction des ratios d'entraînement et de test.
            cls_train_size = int(0.8 * len(cls_indices))
            train_indices.extend(cls_indices[:cls_train_size])
            test_indices.extend(cls_indices[cls_train_size:])

        # Créez des sous-ensembles d'entraînement et de test en utilisant ces indices.
        self.train_dataset = Subset(video_dataset, train_indices)
        self.test_dataset = Subset(video_dataset, test_indices)

    def train_dataloader(self):
        """
        Créer le DataLoader pour le partition d'entraînement
        """
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            shuffle=True,
        )

    def val_dataloader(self):
        """
        Créer le DataLoader pour le partition de validation
        """
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
            shuffle=False,
        )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def uniform_frame_sampling(video_tensor, target_frames=90):
    """
    Sélectionne un nombre uniformément distribué de frames pour que toutes les vidéos aient la même taille.

    Args:
        video_tensor (torch.Tensor): La vidéo originale de forme (T, C, H, W)
        target_frames (int): Nombre de frames cibles à obtenir

    Returns:
        torch.Tensor: Vidéo échantillonnée de forme (target_frames, C, H, W)
    """

    # Obtenir le nombre actuel de frames
    current_frames = video_tensor.shape[0]

    # Si la vidéo actuelle a exactement le nombre cible de frames, la renvoyer telle quelle
    if current_frames == target_frames:
        return video_tensor

    # Calculer les indices des frames à échantillonner
    indices = torch.linspace(0, current_frames - 1, target_frames).long()

    return video_tensor[indices]

# Pour l'utiliser dans le VideoDataset :


def video_resize_and_sample(video_tensor, size=(8, 16), T=90):
    # Resize
    transform = transforms.Resize(size, antialias=False)

    resized_video = [transform(frame) for frame in video_tensor]
    resized_video_tensor = torch.stack(resized_video)
    
    # Uniform sampling
    sampled_video_tensor = uniform_frame_sampling(resized_video_tensor, T)
    
    # Normalisation entre [-1, 1]
    normalized_tensor = (sampled_video_tensor / 127.5) - 1.0

    return normalized_tensor.to(device)
