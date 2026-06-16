# policy_evaluator.py
import torch
import numpy as np

class PrecisionSafetyEvaluator:
    """
    Classe deputata all'applicazione delle metriche di sicurezza clinica,
    calcolando i centroidi di rischio e applicando la policy decisionale.
    """
    def __init__(self, model, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.danger_centroid = None

    def calculate_danger_centroid(self, train_features: np.ndarray, train_labels: np.ndarray):
        """
        Trova la posizione media nello spazio latente dei pazienti tossici/inefficaci del passato.
        """
        self.model.eval() # Modalità valutazione (disattiva dropout)
        
        # Identifichiamo i pazienti storici ad alto rischio
        danger_patients = train_features[train_labels == 1]
        t_danger_patients = torch.FloatTensor(danger_patients).to(self.device)
        
        with torch.no_grad(): # Disattiva il calcolo dei gradienti per risparmiare memoria RAM e velocizzare
            # Embedding dei pazienti ad alto rischio
            danger_embeddings = self.model.forward_once(t_danger_patients)
            
            # Centroide calcolato con media aritmetic
            self.danger_centroid = danger_embeddings.mean(dim=0, keepdim=True)
            
        print(f"Centroide di pericolo calcolato con successo su {danger_patients.shape[0]} pazienti storici.")

    def evaluate_exclusion_policy(self, test_features: np.ndarray, radius_threshold: float = 0.8):
        """
        Prende nuovi pazienti e decide chi escludere in base alla distanza dal pericolo.
        """
        if self.danger_centroid is None:
            raise ValueError("Impossibile valutare la policy senza aver prima calcolato il centroide di pericolo.")
            
        self.model.eval()
        t_test = torch.FloatTensor(test_features).to(self.device)
        
        with torch.no_grad():
            # Proiettiamo i nuovi pazienti nello spazio latente siamese
            test_embeddings = self.model.forward_once(t_test)
            
            # Calcoliamo la distanza euclidea di ciascun nuovo paziente dal centroide tossico
            distances = torch.nn.functional.pairwise_distance(test_embeddings, self.danger_centroid)
            distances_np = distances.cpu().numpy()
            
        # POLICY SAFETY-FIRST: Se la distanza dal centroide tossico è MINORE della soglia (raggio critico), allora consideriamo il paziente come "Unsafe" e lo escludiamo dal trattamento.
        # il paziente è troppo vicino all'area di pericolo. Dobbiamo ESCLUDERLO (Non Trattare -> True)
        exclusion_decision = distances_np < radius_threshold
        
        return exclusion_decision, distances_np