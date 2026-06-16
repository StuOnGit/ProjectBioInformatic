from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Codice preso dal dataset per importarli e per avere il numero id
# fetch dataset
aids_clinical_trials_group_study_175 = fetch_ucirepo(id=890)

# data (as pandas dataframes)
X = aids_clinical_trials_group_study_175.data.features
y = aids_clinical_trials_group_study_175.data.targets

# metadata
print(aids_clinical_trials_group_study_175.metadata)

# variable information
print(aids_clinical_trials_group_study_175.variables)


class DatasetManager:
    """
    Classe responsabile del recupero, della pulizia, dello scaling 
    e della separazione dei dati in Train e Test set.
    """
    def __init__(self, dataset_id: int = 890):
        self.dataset_id = dataset_id
        self.scaler = StandardScaler()
        self.features_names = None
        
        # Variabili che conterranno i dati grezzi e processati
        self.X_scaled = None
        self.labels = None
        self.treatments = None

    def load_and_preprocess(self):
        """
        Scarica il dataset, gestisce i valori mancanti, definisce la condizione
        di 'Unsafe' (Tossicità/Inefficacia) ed effettua lo scaling delle feature.
        """
        # Ingestion dei dati tramite l'API dell'UCI Repository
        dataset = fetch_ucirepo(id=self.dataset_id)
        X_raw = dataset.data.features
        y_raw = dataset.data.targets
        
        # Uniamo X e y in un unico DataFrame per evitare disallineamenti di indici
        df_completo = pd.concat([X_raw, y_raw], axis=1)
        
        # Nomi minuscoli
        df_completo.columns = df_completo.columns.str.lower()
        
        # Gestione dei Missing Values immediata: sostituzionee con la mediana
        df_clean = df_completo.fillna(df_completo.median())
        
        # 1. Estraiamo la variabile del trattamento (treat: 0 o 1)
        self.treatments = df_clean['treat'].values
        
        # 2. Definizione dell'endpoint composito di "Pericolo" (Precision Safety)
        # Un paziente è 'Unsafe' (1) se la terapia è fallita clinicamente (cid == 1)
        # OPPURE se ha dovuto sospendere la terapia precocemente (offtrt == 1).
        cid_vector = df_clean['cid'].values
        offtrt_vector = df_clean['offtrt'].values
        
        # L'operatore '|' esegue l'OR logico elemento per elemento (Element-wise)
        self.labels = ((cid_vector == 1) | (offtrt_vector == 1)).astype(int)
        
        # 3. Isolamento delle sole caratteristiche biologiche e cliniche del paziente (Baseline Features)
        # Dobbiamo assolutamente rimuovere le variabili che indicano il futuro o l'esito (Data Leakage)
        # Rimuoviamo: 
        # - 'treat' e 'trt' (le scelte di trattamento)
        # - 'cid' e 'offtrt' (le risposte che la rete deve invece imparare a evitare indirettamente)
        # - 'time' (il tempo di follow-up, che è legato direttamente all'esito)
        colonne_da_escludere = ['treat', 'trt', 'cid', 'offtrt', 'time']
        
        X_features = df_clean.drop(columns=colonne_da_escludere)
        self.features_names = X_features.columns.tolist()
        
        # Normalizzazione Z-score finale sulle feature per stabilizzare i gradienti di PyTorch
        self.X_scaled = self.scaler.fit_transform(X_features)
        
        print("\n[INFO - DatasetManager] Preprocessing completato con successo.")
        print(f"-> Numero di Feature cliniche di base utilizzate: {len(self.features_names)}")
        print(f"-> Feature incluse: {self.features_names}")
        print(f"-> Totale pazienti 'Unsafe' (da evitare) identificati: {np.sum(self.labels)} su {len(self.labels)}")    
    def get_train_test_splits(self, test_size: float = 0.2, random_state: int = 42):
        """
        Suddivide il dataset in addestramento e test, garantendo la riproducibilità tramite seed.
        """
        if self.X_scaled is None:
            raise ValueError("I dati non sono ancora stati caricati. Chiama prima load_and_preprocess().")
            
        # Effettuiamo uno split stratificato sulle etichette per mantenere la stessa proporzione
        # di pazienti sani/tossici sia nel train che nel test set
        # In pratica garantiamo che la distribuzione di "Unsafe" sia simile per i set
        return train_test_split(
            self.X_scaled, 
            self.labels, 
            self.treatments,
            test_size=test_size, 
            random_state=random_state,
            stratify=self.labels
        )