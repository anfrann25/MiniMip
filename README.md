# Exaflow — Staleness-Aware Federated Learning

> Communication-efficient Federated Learning με μηχανισμούς caching παραμέτρων και staleness-aware επικοινωνία.

Πτυχιακή εργασία: **Staleness-Aware Federated Learning** (Ομοσπονδιακή Μάθηση με Επίγνωση της Παλαιότητας των Ενημερώσεων)
Πρόγραμμα Προπτυχιακών Σπουδών, ΕΚΠΑ — Φεβρουάριος 2026

---

## 📌 Περιγραφή

Το **Exaflow** είναι ένα framework Federated Learning βασισμένο σε αρχιτεκτονική client–server με **gRPC**, το οποίο στοχεύει στη μείωση του κόστους επικοινωνίας μεταξύ πελατών και κεντρικού διακομιστή, χωρίς να επηρεάζεται αρνητικά η ακρίβεια των εκπαιδευόμενων μοντέλων.

Η προσέγγιση βασίζεται σε δύο βασικούς άξονες:

1. **Model Caching & Adaptive Reuse** — κάθε client διατηρεί τοπική cache των παγκόσμιων παραμέτρων και συνεχίζει την τοπική εκπαίδευση («chasing» στρατηγική) χωρίς να απαιτείται άμεσος συγχρονισμός σε κάθε γύρο.
2. **Staleness-Aware Communication** — εμπνευσμένο από το πλαίσιο **SANCUS** (staleness-aware communication-avoiding training), το σύστημα χρησιμοποιεί ένα *bounded staleness* κριτήριο (`τ_max`) σε συνδυασμό με ένα κριτήριο απόκλισης βαρών (`ε_w`) για να αποφασίζει δυναμικά πότε είναι πραγματικά απαραίτητος ο συγχρονισμός.

Αναπτύχθηκαν και συγκρίθηκαν δύο παραλλαγές:

| Υλοποίηση | Περιγραφή |
|---|---|
| **Baseline (`FederatedLogisticRegressionClientSaSo`)** | Κλασικό, πλήρως συγχρονισμένο FedAvg — αποστολή πλήρους payload σε κάθε γύρο. |
| **Proposed / Cached (`FederatedLogisticRegressionClientCached`)** | Staleness-aware caching, με επιλεκτικό συγχρονισμό (`FULL_UPDATE` vs `EMPTY_UPDATE`). |

---

## 🧠 Πώς λειτουργεί

Ο πυρήνας της λογικής βρίσκεται στη μέθοδο `_should_aggregate()`, η οποία αποφασίζει αν ένας γύρος επικοινωνίας είναι απαραίτητος:

```python
def _should_aggregate(self):
    if self.round_counter <= self.warmup_rounds:
        return True

    # Tau-based bound (temporal staleness)
    _, t_coef = self.cache["coef"]
    if (self.round_counter - t_coef) > self.tau_max:
        return True

    # Variation-gap bound (L2 Norm divergence)
    dcoef = np.linalg.norm(self.model.coef_ - self.prev_coef)
    dint = np.linalg.norm(self.model.intercept_ - self.prev_intercept)

    return (dcoef + dint) > self.eps_w
```

- **Warmup phase** — οι πρώτοι γύροι συγχρονίζονται πάντα, ώστε να υπάρχει σταθερή αρχική βάση.
- **Temporal bound (`τ_max`)** — αν η cache έχει «γεράσει» πέρα από το όριο, επιβάλλεται `FULL_UPDATE`.
- **Weight divergence (`ε_w`)** — αν οι τοπικές παράμετροι δεν έχουν μεταβληθεί σημαντικά (Ευκλείδεια νόρμα L2), ο γύρος γίνεται `EMPTY_UPDATE` (ελαφρύ heartbeat αντί για πλήρες tensor payload).

Όταν ένας γύρος παραλείπεται, ο client δεν μένει ανενεργός — αναμειγνύει τοπικά βάρη με τα cached global μέσω convex combination (`_mix_with_cached()`):

```
w_next = η_local · w_local + η_cached · w_cached
```

Η συνάρτηση συγκέντρωσης στον server ενσωματώνει επίσης έναν παράγοντα απόσβεσης παλαιότητας `α(s_k) = 1 / (1 + γ·(t - τ_k))`, ώστε οι πιο «μπαγιάτικες» ενημερώσεις να συνεισφέρουν λιγότερο στο global μοντέλο.

---

## 🏗️ Αρχιτεκτονική

```
┌─────────────┐        gRPC (protobuf)        ┌──────────────┐
│   Client 1  │ ─────────────────────────────▶ │              │
│ (SAGA / LR) │ ◀───────────────────────────── │    Server    │
└─────────────┘                                │ (Aggregator) │
┌─────────────┐                                │              │
│   Client 2  │ ─────────────────────────────▶ │              │
│   ...       │ ◀───────────────────────────── │              │
└─────────────┘                                └──────────────┘
```

- **Μοντέλο:** Scikit-Learn `LogisticRegression` με τον solver **SAGA**.
- **Επικοινωνία:** gRPC / HTTP2, με serialization NumPy arrays σε protobuf.
- **Παραλληλισμός:** Python `multiprocessing` (ξεχωριστές διεργασίες για server & clients, `Manager().dict()` για IPC).
- **Ενορχήστρωση πειραμάτων:** `AutomateTest.py` — αυτοματοποιημένη εκτέλεση, συλλογή μετρικών, παραγωγή γραφημάτων.

---

## 📊 Datasets

| Dataset | Μέγεθος | Domain |
|---|---|---|
| Heart Disease | 35 KB | Κλινική διάγνωση |
| Water Quality | 525 KB | Περιβαλλοντική ποιότητα |
| ICS3D (IoT Cybersecurity) | 82 MB | Βιομηχανικό IoT / εισβολές |
| Lung Cancer | 93 MB | Πρόβλεψη επιβίωσης ασθενών |

Ο διαχωρισμός δεδομένων ανά client γίνεται με `StratifiedKFold`, ώστε όλοι οι clients να έχουν ισορροπημένη κατανομή κλάσεων (Non-IID σενάρια εκτός scope — βλ. Future Work).

---

## ⚙️ Βασικές παράμετροι

| Παράμετρος | Περιγραφή | Default |
|---|---|---|
| `τ_max` | Μέγιστη ανεκτή παλαιότητα cache (γύροι) | 20 |
| `ε_w` | Όριο απόκλισης βαρών (L2) | 0.1 |
| `warmup_rounds` | Γύροι υποχρεωτικού συγχρονισμού στην αρχή | 2 |
| `aggregation_interval` | Fallback περιοδικό διάστημα συγχρονισμού | 3 |
| `η_local` | Συντελεστής μείξης τοπικού μοντέλου | 0.7 |
| `η_cached` | Συντελεστής μείξης cached μοντέλου | 0.3 |
| `T` | Συνολικοί γύροι εκπαίδευσης | 100 |

---

## 📈 Αποτελέσματα

Συγκριτική αξιολόγηση Baseline (S) vs Exaflow/Cached (C) σε 100 γύρους, 3 clients:

| Dataset | Μέγεθος | Acc. (S) | Acc. (C) | Χρόνος (S) | Χρόνος (C) | Μείωση χρόνου |
|---|---|---|---|---|---|---|
| Heart Disease | 35 KB | 0.815 | 0.815 | 6.4 s | 6.2 s | 2.6% |
| Water Quality | 525 KB | 0.628 | 0.628 | 7.4 s | 4.2 s | 43.6% |
| Lung Cancer | 93 MB | 0.779 | 0.779 | 3.46M ms | 1.52M ms | 56.1% |
| IoT Cybersec. | 82 MB | 0.859 | 0.858 | 2.49M ms | 1.19M ms | 52.1% |

**Communication overhead:** σε όλα τα datasets, περίπου **94% των γύρων** μετατρέπονται από `FULL_UPDATE` σε `EMPTY_UPDATE` (18 πλήρεις ενημερώσεις έναντι 282 heartbeats, ανά 300 round-decisions).

**Βασικά συμπεράσματα:**
- Διατήρηση ακρίβειας σχεδόν πανομοιότυπης με το baseline σε όλα τα datasets.
- Το κέρδος στον χρόνο εκτέλεσης κλιμακώνεται με το μέγεθος δεδομένων/μοντέλου — αμελητέο σε μικρά datasets, >50% σε datasets κλίμακας δεκάδων MB.
- Υπό τις προεπιλεγμένες τιμές, το `τ_max` (όχι το `ε_w`) είναι αυτό που κυρίως καθορίζει το πότε γίνεται πλήρης συγχρονισμός.

---

## 🗂️ Δομή repo (ενδεικτική)

```
.
├── thesis.tex                  # Κείμενο πτυχιακής (LaTeX, class: dimscthesis)
├── references.bib
├── figures/                    # Εικόνες/διαγράμματα thesis
├── experiment_results/         # Αποτελέσματα πειραμάτων ανά dataset
│   ├── heart/
│   ├── water/
│   ├── lung/
│   └── ics3d/
├── src/
│   ├── client_baseline.py      # FederatedLogisticRegressionClientSaSo
│   ├── client_cached.py        # FederatedLogisticRegressionClientCached
│   ├── server.py                # gRPC Aggregator
│   └── AutomateTest.py          # Ενορχήστρωση πειραμάτων
└── README.md
```

> Προσάρμοσε τη δομή παραπάνω ώστε να αντιστοιχεί ακριβώς στα πραγματικά path/ονόματα αρχείων του repo σου.

---

## 🔬 Software Stack

- **Python 3.x**
- **Scikit-Learn** (`LogisticRegression`, SAGA solver)
- **NumPy** (L2 norms, πράξεις πινάκων)
- **gRPC** (μεταφορά serialized παραμέτρων)
- **Python multiprocessing** (προσομοίωση κατανεμημένου περιβάλλοντος)

---

## ⚠️ Περιορισμοί

- Η αξιολόγηση έγινε σε **στρωματοποιημένη (IID-like)** κατανομή δεδομένων· η συμπεριφορά υπό έντονο Non-IID skew δεν έχει μετρηθεί.
- Χρησιμοποιήθηκε **Logistic Regression** (μικρός, κυρτός χώρος παραμέτρων) — η γενίκευση σε νευρωνικά δίκτυα (μη κυρτό) παραμένει ανοιχτό ζήτημα.
- Η τοπολογία προσομοιώθηκε ως ξεχωριστές διεργασίες σε **μία φυσική μηχανή** (όχι πραγματικό WAN).
- Ο server υλοποιεί **synchronous barrier** — αληθινά ανεξάρτητο (ασύγχρονο) skipping ανά client θα απαιτούσε επέκταση του aggregation protocol (βλ. Future Work).

## 🔭 Μελλοντική εργασία

- Aggregation server χωρίς synchronous barrier (κλείσιμο γύρου μόνο με τους non-skipping clients).
- Συστηματική μελέτη υπό ελεγχόμενο Non-IID skew.
- Επέκταση σε νευρωνικά δίκτυα.
- Συνδυασμός με τεχνικές συμπίεσης payload (quantization/sparsification).
- Αξιολόγηση σε πραγματικό WAN για μέτρηση end-to-end latency.

---

## 📚 Βασικές αναφορές

- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data* (FedAvg)
- Li et al., *FedProx: Federated Optimization in Heterogeneous Networks*
- Peng et al., *SANCUS: Staleness-Aware Communication-Avoiding Full-Graph Decentralized Training*
- Defazio et al., *SAGA: A Fast Incremental Gradient Method*
