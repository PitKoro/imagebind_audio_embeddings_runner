from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from typing import List

class ImageBindModel:
    def __init__(self):
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def get_audio_embeddings_by_audio_path(self, audio_paths: List[str]) -> List[List[float]]:
        inputs = {
            ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, self.device),
        }
        with torch.no_grad():
            embeddings = self.model(inputs)
        if self.device != 'cpu':
            embeddings = embeddings[ModalityType.AUDIO].detach().cpu().numpy()
            torch.cuda.empty_cache()
        return embeddings

