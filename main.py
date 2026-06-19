# main.py
import torch
from torch.utils.data import DataLoader
from dataset_manager import DatasetManager
from siamese_dataset import SiamesePatientDataset
from network import SiameseTwinNet
from losses import ContrastiveLoss
from trainer import SiameseTrainer
from policy_evaluator import PrecisionSafetyEvaluator
from metrics import (
    build_siamese_pairs,
    compute_pairwise_distances,
    select_best_distance_threshold,
    compute_pairwise_metrics,
    compute_policy_safety_metrics,
)
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
    # Coso per ottimizzare dato che potrebbe essere lento
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    
    # Roba per velocizzare e ottimizzare (CPU o GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = SiameseTrainer(model, criterion, optimizer, device=device)
    
    print("\n=== Fase 4: Addestramento del Modello (Spazio di Contrasto) ===")
    epochs = 1000
    for epoch in range(1, epochs + 1):
        loss = trainer.train_epoch(train_loader)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoca [{epoch}/{epochs}] -> Contrastive Loss [Media]: {loss:.4f}")
            
    print("=== Fase 5: Valutazione del Modello Siamese ===")
    eval_x1, eval_x2, eval_labels = build_siamese_pairs(X_test, y_test, num_pairs=2000, seed=42)
    distances = compute_pairwise_distances(model, eval_x1, eval_x2, device=device)
    best_threshold = select_best_distance_threshold(distances, eval_labels)
    pairwise_metrics = compute_pairwise_metrics(distances, eval_labels, threshold=best_threshold)

    print("\n=== METRICHE SIAMESE SULLA DISTANZA DI COPPIA ===")
    print(f"Soglia ottimale di distanza: {pairwise_metrics['threshold']:.4f}")
    print(f"Accuracy: {pairwise_metrics['accuracy']:.4f}")
    print(f"Precision: {pairwise_metrics['precision']:.4f}")
    print(f"Recall: {pairwise_metrics['recall']:.4f}")
    print(f"F1-score: {pairwise_metrics['f1']:.4f}")
    print(f"ROC-AUC: {pairwise_metrics['roc_auc']:.4f}")
    print(f"Average Precision: {pairwise_metrics['average_precision']:.4f}")
    print(f"Media distanza coppie simili: {pairwise_metrics['mean_distance_similar']:.4f}")
    print(f"Media distanza coppie diverse: {pairwise_metrics['mean_distance_different']:.4f}")
    print(f"Confusion Matrix:{pairwise_metrics['confusion_matrix'] if pairwise_metrics['confusion_matrix'] is not None else 'Non disponibile'}")
    
    cm = pairwise_metrics.get('confusion_matrix')
    if cm is not None:
        print("\nConfusion Matrix (righe=true, colonne=predette):")
        print(np.array(cm))

    print("\n=== Fase 6: Estrazione Policy di Precision Safety ===")
    evaluator = PrecisionSafetyEvaluator(model, device=device)
    evaluator.calculate_danger_centroid(X_train, y_train)
    soglia_raggio = 0.75
    decisioni_esclusione, distanze = evaluator.evaluate_exclusion_policy(X_test, radius_threshold=soglia_raggio)

    policy_metrics = compute_policy_safety_metrics(y_test, decisioni_esclusione)
    percentuale_esclusi = policy_metrics['exclusion_rate'] * 100

    print(f"\n=== RISULTATI DELLA POLICY SUL TEST SET ===")
    print(f"Percentuale di pazienti ESCLUSI dal trattamento per motivi di sicurezza: {percentuale_esclusi:.2f}%")
    print(f"Sensibilità del Framework (True Positive Rate): {policy_metrics['sensitivity']:.4f}")
    print(f"Specificità della policy: {policy_metrics['specificity']:.4f}")
    print(f"Precision della policy: {policy_metrics['precision']:.4f}")
    print(f"False Positive Rate: {policy_metrics['false_positive_rate']:.4f}")
    print(f"False Negative Rate: {policy_metrics['false_negative_rate']:.4f}")

if __name__ == "__main__":
    run_precision_safety_experiment()