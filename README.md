# ProjectBioInformatic

# Esempi
* ## Uber: Ottimizzazione delle Promozioni e Ritenzione degli Utenti
  **Uber** deve decidere se e quanto sconto (trattamento) offrire a un utente per incentivarlo a usare l'app dopo che ha vissuto un'esperienza negativa (es. un forte ritardo del conducente). Dare sconti a tutti è costosissimo, mentre darne troppi pochi causa l'abbandono dell'utente (churn).
  ### L'Applicazione Causale MTL
  I data scientist di Uber hanno implementato modelli di Uplift Modeling basati su reti neurali Multi-Task. Nello specifico, hanno integrato e reso open-source l'architettura [DragonNet](https://towardsdatascience.com/tarnet-and-dragonnet-causal-inference-between-s-and-t-learners-0444b8cc65bd/). La rete ha una base condivisa che apprende una rappresentazione dell'utente (cronologia viaggi, zona, livello di insoddisfazione) e si dirama in task/teste diverse per prevedere l'esito (se l'utente abbandona o meno) con o senza la promozione, calcolando l'effetto causale netto condizionale (CATE).
  ### Fonti
  - [Paper scientifico](https://proceedings.mlr.press/v104/du19a/du19a.pdf)
   - [Codice Sorgente](https://gemini.google.com/app/ee3024598ce79bf1#:~:text=Vedi%20il%20codice%20di%20DragonNet%20su%20GitHub%20)
* ## Booking: Valutare l'Impatto Reale delle Nuove Funzionalità (Voluntary Adoption)
  Quando Booking rilascia una nuova funzionalità (ad esempio, un nuovo sistema di pagamento flessibile opzionale per gli hotel partner), non può obbligare nessuno a usarla. Gli hotel che decidono autonomamente di adottarla sono solitamente quelli più moderni, grandi o digitalizzati. Se Booking confrontasse semplicemente i partner che usano la feature con quelli che non la usano, vedrebbe che chi la usa guadagna di più. Ma è merito della feature (Causa) o del fatto che quegli hotel erano già più forti in partenza (Confondimento/Bias di selezione)?
  ### L'Applicazione Causale MTL
  **Booking**  utilizza il [Machine Learning Causale](https://arxiv.org/abs/2206.15475) e architetture neurali **Multi-Task** per stimare l'impatto reale isolando il bias. Tramite un approccio controfattuale, il modello stima contemporaneamente due task (i potenziali outcome): l'andamento delle prenotazioni dell'hotel se decidesse di adottare la feature e l'andamento se decidesse di non farlo, bilanciando le caratteristiche di partenza dei partner per simulare un perfetto "A/B test" anche dove l'esperimento puro è impossibile.
  ### Fonti
  - [Causality Gap](https://booking.ai/the-causality-gap-measuring-the-true-impact-of-voluntary-adoption-in-digital-marketplaces-ea68b5a35120)
  * ## Piattaforme di Streaming e E-Commerce (es. Netflix, Alibaba): Uplift Modeling Multi-Trattamento per le Raccomandazioni
  Nelle moderne piattaforme di e-commerce o streaming, non esiste un solo trattamento binario (Sconto Sì / Sconto No), ma esistono trattamenti multipli (es. Mostrare la locandina A, la locandina B, inviare una **notifica push** aggressiva o una soft). L'obiettivo è massimizzare l'engagement totale minimizzando il disturbo all'utente.
  ### L'Applicazione Causale MTL
  In questi contesti si usano estensioni di **TARNet** e modelli Multi-Task in cui la rete neurale non ha solo 2 teste, ma ha "N" teste (una per ogni possibile trattamento/incentivo disponibile).Lo strato condiviso estrae i pattern storici dell'utente. Ogni "testa" predice l'esito specifico sotto quel preciso trattamento (es. la probabilità di cliccare sul video vedendo la locandina X).Questo approccio, ampiamente documentato nelle conferenze di settore come [KDD](https://www.isti.cnr.it/it/ricerca/ricerca/laboratories/9/Knowledge_Discovery_and_Data_Mining_KDD)(Knowledge Database and Data Mining) o [RecSys](https://recsys.acm.org/), permette di fare decisioni in tempo reale (**Online Dynamic Decision-Making**) calcolando quale specifico trattamento genererà il maggior incremento (**uplift**) di conversioni rispetto allo scenario di controllo (nessun intervento).

  ### Fonti
  - [Paper di Ricerca sull'Uplift Multi-Trattamento](https://www.researchgate.net/publication/338797004_Uplift_Modeling_for_Multiple_Treatments_with_Cost_Optimization)
   - [tarNet e arXiv](https://arxiv.org/abs/1606.03976)