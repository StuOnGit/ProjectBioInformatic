# losses.py
import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss per ottimizzare le distanze euclidee nello spazio latente / presa da github - da rivedere
    """
    def __init__(self, margin: float = 1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, embedding1: torch.Tensor, embedding2: torch.Tensor, label: torch.Tensor):
        # Calcolo della distanza euclidea punto a punto tra le due proiezioni
        # eps (epsilon) previene instabilità numeriche se la distanza è esattamente zero
        euclidean_distance = torch.nn.functional.pairwise_distance(embedding1, embedding2, eps=1e-6)
        
        # Componente per coppie Simili: minimizza la distanza al quadrato
        loss_similar = label.squeeze() * torch.pow(euclidean_distance, 2)
        
        # Componente per coppie Diverse: massimizza la distanza spingendola oltre il margine
        loss_dissimilar = (1.0 - label.squeeze()) * torch.pow(
            torch.clamp(self.margin - euclidean_distance, min=0.0), 2
        )
        
        # Calcolo della media complessiva della perdita per il batch corrente
        loss_contrastive = torch.mean(loss_similar + loss_dissimilar)
        return loss_contrastive