# network.py
import torch
import torch.nn as nn

class SiameseTwinNet(nn.Module):
    """
    Architettura Siamese con una singola sotto-rete per l'estrazione di embedding.
    """
    def __init__(self, input_dim: int, embedding_dim: int = 16):
        super(SiameseTwinNet, self).__init__()
        
        # Architettura sequenziale della sub-network (il 'Gemello')
        self.sub_network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64), # Stabilizza l'addestramento normalizzando gli output intermedi dello strato
            nn.Dropout(0.2),    # Regolarizzazione: spegne casualmente il 20% dei neuroni per evitare overfitting
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            
            nn.Linear(32, embedding_dim) # Lo strato finale mappa il paziente nello spazio latente compresso
        )
        
    def forward_once(self, x):
        # Un solo passaggio per ottenere l'embedding di un singolo paziente
        return self.sub_network(x)
        
    def forward(self, x1, x2):
        # Qua la passa a entrambi (dato che è siamese la rete)
        embedding1 = self.forward_once(x1)
        embedding2 = self.forward_once(x2)
        return embedding1, embedding2