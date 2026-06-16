# siamese_dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np

class SiamesePatientDataset(Dataset):
    """
    Estensione di PyTorch Dataset per generare coppie di pazienti in tempo reale.
    """
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = features
        self.labels = labels
        self.num_samples = len(self.labels)

    def __len__(self):
        # Il numero di campioni totali a disposizione in un'epoca
        return self.num_samples

    def __getitem__(self, idx: int):
        """
        Funzione core richiamata dal DataLoader di PyTorch. 
        Dato un indice, estrae un paziente e ne seleziona un secondo per creare la coppia.
        """
        x1 = self.features[idx]
        label1 = self.labels[idx]
        
        # Decidiamo se creare una coppia SIMILE (stesso esito) o DIVERSA (esito opposto)
        # Questo bilanciamento 50/50 impedisce alla rete di sbilanciarsi verso una sola risposta.
        should_get_same_class = np.random.randint(0, 2)
        
        if should_get_same_class:
            # Trova gli indici dei pazienti che hanno la STESSA etichetta di label1
            same_class_indices = np.where(self.labels == label1)[0]
            # Seleziona un indice a caso tra questi
            random_idx = np.random.choice(same_class_indices)
        else:
            # Trova gli indici dei pazienti che hanno un'etichetta DIVERSA da label1
            different_class_indices = np.where(self.labels != label1)[0]
            random_idx = np.random.choice(different_class_indices)
            
        x2 = self.features[random_idx]
        label2 = self.labels[random_idx]
        
        # Se le due label originali sono uguali, la similarità della coppia è 1.0, altrimenti è 0.0
        similarity_label = 1.0 if label1 == label2 else 0.0
        
        # Restituiamo i tensori PyTorch pronti per l'elaborazione su GPU/CPU
        return (
            torch.FloatTensor(x1), 
            torch.FloatTensor(x2), 
            torch.FloatTensor([similarity_label])
        )