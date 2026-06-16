# trainer.py
import torch
from torch.utils.data import DataLoader

class SiameseTrainer:
    """
    Gestore dei cicli di addestramento (Training Loops) e di ottimizzazione dei gradienti.
    """
    def __init__(self, model, criterion, optimizer, device: str = "cpu"):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, dataloader: DataLoader):
        """
        Esegue una singola epoca di addestramento scorrendo tutti i batch del dataloader.
        """
        self.model.train() # Imposta il modello in modalità addestramento (attiva Dropout e BatchNorm)
        running_loss = 0.0
        
        for batch_x1, batch_x2, batch_y in dataloader:
            # Sposta i batch o sulla gpu o sulla cpu (ottimizzazione)
            batch_x1 = batch_x1.to(self.device)
            batch_x2 = batch_x2.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Reset dei gradienti accumulati nel passo precedente
            self.optimizer.zero_grad()
            
            # Forward pass: calcolo degli embedding
            emb1, emb2 = self.model(batch_x1, batch_x2)
            
            # Calcolo dell'errore tramite la Contrastive Loss
            loss = self.criterion(emb1, emb2, batch_y)
            
            # Backward pass: calcolo dei gradienti tramite catena di back propagation
            loss.backward()
            
            # Aggiornamento dei pesi 
            self.optimizer.step()
            
            running_loss += loss.item() * batch_x1.size(0)
            
        epoch_loss = running_loss / len(dataloader.dataset)
        return epoch_loss