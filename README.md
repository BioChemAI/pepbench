[readme.md](https://github.com/user-attachments/files/22362435/readme.md)
# Peptide Property Prediction
## 📁Project Structure
~~~
pepbench/
├── data/                      # Data storage directory
│   ├── Binary_Classification/ # Binary classification task data               
│   └── Regression/            # Regression task data
│
├── feature/                   # Feature engineering module
│   ├── onehot.py              # One-Hot encoding
│   └── descriptor.py          # Molecular descriptor encoding
│     
├── MODEL/                     # Pretrained model storage               
│   ├── prot_bert/             # PepBERT model
│   └── esm2_t12_35M_UR50D/    # ESM model
│
├── model/                     # Model architecture definitions
│   ├── base.py                #
│   ├── factory.py             # 
│   ├── rf.py                  # Random Forest model
│   ├── svm.py                 # Support Vector Machine model
│   ├── xgb.py                 # XGBoost model
│   ├── lstm.py                # LSTM model
│   ├── transformer.py         # Transformer model
│   ├── pepbert.py             # PepBERT model
│   └── esm.py                 # ESM model
│
├── utils/                     # Utility functions  
│   └── metrics.py             # Evaluation metrics
│                      
├── saved_models/              # Path for saving trained models     
├── train.py                   # Main training script           
├── dataset.py                 # Data loader
├── model_manager.py           # Model management tool
├── test.py                    # Testing and evaluation script
└── environment.yml            # Project dependencies
~~~      

## 📊Data Introduction
### Data Sources
The peptide data used in this project comes from public databases and experimental measurements, including binary classification datasets and regression datasets:
#### Binary Classification Datasets
##### 1.Antidiabetic Peptide(ADP)
- **Source**：*Discovery of potential antidiabetic peptides using deep learning*
- **Positive samples**：418
- **Negative samples**：5250
- **Length range**：4-99
- **Description**：Contains only natural amino acids

##### 2.Antimicrobial Peptide(amp_50)
- **Source**：Positive samples were integrated from the APD3, DBAASP, and DRAMP databases, retaining only sequences with both N- and C-termini being free or empty, followed by merging and deduplication. Negative samples were collected from the UniProt database by applying the “subcellular location” filter set to “cytoplasm,” with sequence length less than 50. Entries containing any of the following keywords were removed: antimicrobial, antibiotic, antiviral, antifungal, effector, excreted. The filtering was performed according to the paper *Identification of antimicrobial peptides from the human gut microbiome using deep learning*.
- **Positive samples**：28756
- **Negative samples**：8003
- **Length range**：1-183
- **Description**：Contains only natural amino acids

##### 3.Antioxidant Peptide(AOPP)
- **Source**：Antioxidant Peptide Prediction database
- **Positive samples**：1586
- **Negative samples**：1578
- **Length range**：2-20
- **Description**：Contains only natural amino acids

##### 4.Self-assembling Peptide(assem)
- **Source**：*Efficient prediction of peptide self-assembly through equential and graphical encoding* and *Reshaping the discovery of self-assembling peptides with generative AI guided by hybrid deep learning*
- **Positive samples**：15007
- **Negative samples**：26697
- **Length range**：3-24
- **Description**：Contains only natural amino acids

##### 5.Blood–Brain Barrier Penetrating Peptide(BBB)
- **Source**：*Improved prediction and characterization of blood-brain barrier penetrating peptides using estimated propensity scores of dipeptides*
- **Positive samples**：265
- **Negative samples**：257
- **Length range**：4-30
- **Description**：Contains only natural amino acids

##### 6.Cell-Penetrating Peptide(CPP)
- **Source**：*StackCPPred: a stacking and pairwise energy  content-based prediction of cell-penetrating peptides  and their uptake efficiency*
- **Positive samples**：462
- **Negative samples**：462
- **Length range**：4-61
- **Description**：Contains only natural amino acids

##### 7.Dipeptidyl Peptidase IV Inhibitory Peptide(DPPIV)
- **Source**：*StackDPPIV: A novel computational approach for accurate prediction of dipeptidyl peptidase IV (DPP-IV) inhibitory peptides*
- **Positive samples**：664
- **Negative samples**：665
- **Length range**：2-90
- **Description**：Contains only natural amino acids

##### 8.Hemolysis Peptide(hemo)
- **Source**：*PeptideBERT: A Language Model Based on Transformers for Peptide Property Prediction*
- **Positive samples**：1826
- **Negative samples**：7490
- **Length range**：1-190
- **Description**：Contains only natural amino acids

##### 9.Nonfouling Peptide(human)
- **Source**：*PeptideBERT: A Language Model Based on Transformers for Peptide Property Prediction*
- **Positive samples**：3600
- **Negative samples**：13585
- **Length range**：4-198
- **Description**：Contains only natural amino acids

##### 10.Neuropeptide(NEU)
- **Source**：*NeuroPred-PLM: an interpretable and robust model for neuropeptide prediction by protein language model*
- **Positive samples**：4393
- **Negative samples**：4306
- **Length range**：4-99
- **Description**：Contains only natural amino acids

##### 11.Solubility Peptide(souble)
- **Source**：*PeptideBERT: A Language Model Based on Transformers for Peptide Property Prediction*
- **Positive samples**：8785
- **Negative samples**：9668
- **Length range**：4-198
- **Description**：Contains only natural amino acids

##### 12.Toxic Peptide(toxic)
- **Source**：ToxinPred2 dataset
- **Positive samples**：1052
- **Negative samples**：464
- **Length range**：1-200
- **Description**：Contains only natural amino acids

#### Regression Datasets
##### 1.EC
- **Source**：*BERT-AmPEP60: A BERT-Based Transfer Learning Approach to Predict the Minimum Inhibitory Concentrations of Antimicrobial Peptides for Escherichia coli and Staphylococcus aureus*
- **Number of samples**：4042
- **Length range**：
- **Description**：Contains only natural amino acids

##### 2.SA
- **Source**：*BERT-AmPEP60: A BERT-Based Transfer Learning Approach to Predict the Minimum Inhibitory Concentrations of Antimicrobial Peptides for Escherichia coli and Staphylococcus aureus*
- **Number of samples**：3275
- **Length range**：
- **Description**：Contains only natural amino acids

##### 3.Cell-Penetrating Peptide
- **Source**：CycPeptMPDB
- **Number of samples**：8466
- **Length range**：
- **Description**：Contains non-natural amino acids

##### 4.Hemolysis Peptide
- **Source**：HemoPI2 - Hemolytic Activity Prediction
- **Number of samples**：1926
- **Length range**：
- **Description**：Contains only natural amino acids

### Data Format
The data is stored in CSV format and contains the following columns:
~~~
,peps,label
1527,FLGAILKIGHALAKTVLPMVTNAFKPKQ,0.0
173,SPLGQSQPTVAGQPSARPAAEEYGYIVTDQKPLSLAAGVK,1.0
1032,QGVRNSQSCRRNKGICVPIRCPGSMRQIGTCLGAQVKCCRRK,5.161810388853155
~~~
### Description of Columns
- **id column**：Serial number, no special meaning
- **peps column**：Peptide sequence represented by amino acid single-letter codes
- **label column**：Indicates whether the peptide has a certain function or the strength of its activity


## 🚀Quick Start
### Install Dependencies
```bash
conda env create -f environment.yml
conda activate pepbench
```
### Dataset Splitting
```bash
python randomsplit.py --data_path data/Binary_Classification/ADP.csv --random_state 111 # Random split
python cdhitsplit.py # Similarity-based split
```
### Feature Extraction
onehot
descriptor
### Train Models
~~~bash
python train.py --task classification --model rf --random_state 111 --train_path data/Binary_Classification/splitter111/ADP_train.csv --val_path data/Binary_Classification/splitter111/ADP_val.csv --max_len 198 --data_name ADP
~~~
### Test Models
~~~bash
python test.py --task classification --model rf --model_path saved_models/svm_classification_ADP_seed111_20250714_173604.pkl --test_path data/Binary_Classification/splitter111/ADP_test.csv --max_len 41
~~~

## 🤝Contribution
## 📧Connection
## 📄License
