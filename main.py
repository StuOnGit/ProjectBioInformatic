# main.py
import torch
from torch.utils.data import DataLoader
from dataset_manager import DatasetManager
from siamese_dataset import SiamesePatientDataset
from network import SiameseTwinNet
from losses import ContrastiveLoss
from trainer import SiameseTrainer
from policy_evaluator import PrecisionSafetyEvaluator
import numpy as np

def run_precision_safety_experiment():
    print("=== Fase 1: Caricamento e Preprocessing Dati (ACTG 175) ===")

    dm = DatasetManager(dataset_id=890)
    dm.load_and_preprocess()
    
    
    # Estraiamo i vettori suddivisi
    X_train, X_test, y_train, y_test, W_train, W_test = dm.get_train_test_splits(test_size=0.2)
    
    print(f"Campioni di Addestramento: {X_train.shape[0]} | Campioni di Test: {X_test.shape[0]}")
    
    print("\n=== Fase 2: Creazione dei Dataset Siamesi a Coppie ===")
    train_dataset = SiamesePatientDataset(X_train, y_train)
    # Il DataLoader gestisce il rimescolamento stocastico e la divisione in piccoli batch per l'addestramento
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    print("\n=== Fase 3: Inizializzazione della Rete Siamese ===")
    input_dimension = X_train.shape[1]
    embedding_dimension = 8 # Dimensione del vettore di embedding (spazio latente) che rappresenta ogni paziente dopo la proiezione
    
    model = SiameseTwinNet(input_dim=input_dimension, embedding_dim=embedding_dimension)
    criterion = ContrastiveLoss(margin=1.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    
    # Roba per velocizzare e ottimizzare (CPU o GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = SiameseTrainer(model, criterion, optimizer, device=device)
    
    print("\n=== Fase 4: Addestramento del Modello (Spazio di Contrasto) ===")
    epochs = 200
    for epoch in range(1, epochs + 1):
        loss = trainer.train_epoch(train_loader)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoca [{epoch}/{epochs}] -> Contrastive Loss [Media]: {loss:.4f}")
            
    print("\n=== Fase 5: Estrazione Policy di Precision Safety ===")
    evaluator = PrecisionSafetyEvaluator(model, device=device)
    # Calcoliamo il centroide basandoci sulla conoscenza clinica pregressa (Train Set)
    evaluator.calculate_danger_centroid(X_train, y_train)
    
    # Eseguiamo la policy sul Test Set (Pazienti simulati futuri)
    # Raggio critico impostato a 0.75 / serve per bilanciare esclusione e trattamento
    soglia_raggio = 0.75 
    decisioni_esclusione, distanze = evaluator.evaluate_exclusion_policy(X_test, radius_threshold=soglia_raggio)
    
    # Calcolo delle statistiche sulla popolazione di test
    percentuale_esclusi = np.mean(decisioni_esclusione) * 100
    print(f"\n=== RISULTATI DELLA POLICY SUL TEST SET ===")
    print(f"Percentuale di pazienti ESCLUSI dal trattamento per motivi di sicurezza: {percentuale_esclusi:.2f}%")

    # Quanti pazienti individuati?    
    tossici_reali = (y_test == 1)
    intercettati = decisioni_esclusione[tossici_reali]
    sensibilita_safety = np.mean(intercettati) * 100
    print(f"Sensibilità del Framework (Percentuale di tossicità reali evitate): {sensibilita_safety:.2f}%")

if __name__ == "__main__":
    run_precision_safety_experiment()