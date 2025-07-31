# GENET: Automatic Literature Analysis and Knowledge Graph Construction for Molecular Mechanisms

## Introduction

**GENET** is an automatic literature analysis platform for extracting, validating, and visualizing molecular regulatory relationships from biomedical literature. GENET integrates named entity recognition (NER) and relation extraction (RE) models, constructing a molecular interaction knowledge graph and providing reliable annotation tools for biomedical research.

## Supplementary Methods

### 1. Development of GENET

* **Dataset:**
  A random subset of 1,788 PubMed abstracts was annotated and split into training, validation, and test sets, based on individual publications.
* **Annotation:**
  Annotation was performed by a single expert using Label Studio. The BIO tagging scheme was adopted for NER, with four categories: genes, signaling pathways, cancer, and biological function. The GENE database (NCBI, [https://www.ncbi.nlm.nih.gov/](https://www.ncbi.nlm.nih.gov/)) was referenced for gene annotation.
* **Relation Types:**
  Five types: promotion, inhibition, upstream, abbreviation, and function.

#### NER Module

* **Framework:**
  Combined [Flair](https://github.com/flairNLP/flair) NLP framework and PubMedBERT-large embeddings.
* **Features:**
  PubMedBERT-large captures biomedical term semantics; Flair provides character-level context.
* **Pipeline:**

  * Word and character-level embeddings are merged.
  * An additional linear layer integrates entity vectors.
  * A Recurrent Neural Network (RNN) captures long-range dependencies.
  * A Conditional Random Field (CRF) layer further improves sequence labeling.

#### RE Module

* **Segmentation:**
  Used `nltk.tokenize` for intelligent sentence splitting.
* **Framework:**
  Used PubMedBERT-large (Huggingface) + [UniRel](https://github.com/SapienzaNLP/relik) for unified triple extraction via interaction maps and transformer-based self-attention.
* **Loss:**
  Binary Cross Entropy Loss.
* **Architecture:**
  Sentence encoding and relation prediction are decoupled, with extra linear layers merging their representations.

#### Model Deployment

* **Optimization:**
  Converted models to static graphs and then to TensorRT; applied int8 quantization.
* **Serving:**
  Deployed using [Triton Inference Server](https://github.com/triton-inference-server/server), achieving <50 ms per abstract.

#### Post-processing

* Integrated NER and RE outputs to ensure both entity and relation uniqueness in the final knowledge graph.
* Entity aliases are normalized using the NCBI GENE database.
* Visualization with [AntV G6](https://g6.antv.vision/en).

---

### 2. Validation and Application

* **Question Templates for RE:**

  * Promotion: Where does {entityA} promote {entityB} occur?
  * Inhibition: Where does {entityA} inhibit {entityB} occur?
  * Upstream: Where does {entityA} act as an upstream regulator of {entityB}?
  * Function: Where does the {entityA} function of {entityB} occur?

* **BPS System Validation:**
  The BPS system was validated in deciphering communication mechanisms between colorectal cancer cells and macrophages using tissue, cell lines, KRAS mutation analysis, gene silencing/overexpression, western blot, qPCR, ELISA, and multiplex immunofluorescence.

---

## Supplementary Results

### Performance Metrics

| Task/Model | Precision | Recall | F1-score |
| ---------- | --------- | ------ | -------- |
| NER        | 94.23%    | 97.72% | 95.94%   |
| RE         | 86.94%    | 90.44% | 88.66%   |
| NER + RE   | 91.63%    | 89.17% | 90.38%   |

**Manual validation (100 relations):**

* GENET recall: **97.00%** (97/100)
* GENIE3: 89.00% (89/100)
* Other seven methods: lower (P < 0.001).

GENET outperforms existing models in accuracy and interpretability.

---

## Supplementary Tables

### Table S1. Comparison with LLMs for Answered Abstracts

| Model           | Accuracy       | Precision      | Recall | F1-score |
| --------------- | -------------- | -------------- | ------ | -------- |
| BPS             | 95.45%         | 95.45%         | 97.35% | 96.39%   |
| DeepSeek-V3     | 76.62%\*\*\*\* | 76.62%\*\*\*\* | 97.52% | 85.82%   |
| Doubao          | 85.71%\*\*     | 85.71%\*\*     | 97.06% | 91.03%   |
| Qwen3-235B-A22B | 87.66%\*       | 87.66%\*       | 95.74% | 91.53%   |
| GPT-4-mini      | 87.01%\*       | 87.01%\*       | 99.26% | 92.73%   |
| Grok3           | 87.01%\*\*     | 87.01%\*       | 95.71% | 91.16%   |

> * p < 0.05, \*\* p < 0.01, \*\*\* p < 0.001, \*\*\*\* p < 0.0001.

### Table S2. Comparison for Abstracts Without Answers

| Model           | Accuracy | Precision | Recall  | F1-score |
| --------------- | -------- | --------- | ------- | -------- |
| BPS             | 100.00%  | 100.00%   | 100.00% | 100.00%  |
| DeepSeek-V3     | 66.67%   | 66.67%    | 100.00% | 100.00%  |
| Doubao          | 50.00%   | 50.00%    | 100.00% | 100.00%  |
| Qwen3-235B-A22B | 16.67%\* | 16.67%\*  | 100.00% | 100.00%  |
| GPT-4-mini      | 16.67%\* | 16.67%\*  | 100.00% | 100.00%  |
| Grok3           | 0%\*     | 0%\*\*    | /       | /        |

---

### Table S3. Antibody Information

| Antibody                            | Serial Number | Company     |
| ----------------------------------- | ------------- | ----------- |
| STAT3 Rabbit pAb                    | A19566        | ABclonal    |
| Phospho-STAT3-S727 Rabbit mAb       | AP0715        | ABclonal    |
| Arginase 1 (ARG-1) Rabbit pAb       | A1847         | ABclonal    |
| iNOS Rabbit pAb                     | A0312         | ABclonal    |
| CD68 Rabbit pAb                     | A13286        | ABclonal    |
| TNF-α Rabbit mAb                    | A24214        | ABclonal    |
| GAPDH Monoclonal antibody           | 60004-1-Ig    | Proteintech |
| HRP-conj. Goat Anti-Rabbit IgG(H+L) | SA00001-2     | Proteintech |

### Table S4. Primer Information

| Gene  | Forward Sequence        | Reverse Sequence       |
| ----- | ----------------------- | ---------------------- |
| STAT3 | CTTTGAGACCGAGGTGTATCACC | GGTCAGCATGTTGTACCACAGG |
| IL-10 | TCTCCGAGATGCCTTCAGCAGA  | TCAGACAAGGCTTGGCAACCCA |
| TNF-α | AGCCCATGTTGTAGCAAAC     | TGAGGTACAGGCCCTCTGA    |
| GAPDH | GTCTCCTCTGACTTCAACAGCG  | ACCACCCTGTTGCTGTAGCCAA |

### Table S5. Patient Characteristics for Validation

| Characteristic                | Value                                                                                  |
| ----------------------------- | -------------------------------------------------------------------------------------- |
| Age, years (SD)               | 67.18 (12.91)                                                                          |
| Sex, n (%)                    | Female: 25 (50%)                                                                       |
|                               | Male: 25 (50%)                                                                         |
| Differentiation status, n (%) | Poor: 3 (6%)                                                                           |
|                               | Moderate: 36 (72%)                                                                     |
|                               | Mod-Poor: 10 (20%)                                                                     |
|                               | Unknown: 1 (2%)                                                                        |
| Pathology, n (%)              | Adenocarcinoma: 45 (90%)<br>Mucinous: 2 (4%)<br>Adenocarcinoma w/ mucin: 3 (6%)        |
| Location, n (%)               | Ascending: 27 (54%)                                                                    |
|                               | Descending: 10 (20%)                                                                   |
|                               | Transverse: 1 (2%)                                                                     |
|                               | Sigmoid: 9 (18%)                                                                       |
|                               | Unknown: 3 (6%)                                                                        |
| Gross type                    | Exophytic: 15 (30%)<br>Ulcerative: 33 (66%)<br>Infiltrating: 1 (2%)<br>Unknown: 1 (2%) |
| T-staging, n (%)              | T2: 2 (4%)<br>T3: 33 (66%)<br>T4a: 15 (30%)                                            |
| SD: standard deviation        |                                                                                        |

---

## Reference

\[1] Tang W, Xu B, Zhao Y, et al. UniRel: Unified Representation and Interaction for Joint Relational Triple Extraction. *EMNLP 2022*; 7087-7099.

---

> **Note**: For details on data preparation, training scripts, or knowledge graph visualization, please see the respective folders and scripts in this repository.

---


