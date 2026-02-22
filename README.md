#  NavaRasaBank
> **The First Large-Scale, Human-Validated Corpus for Computational Rasa Analysis in Sanskrit.**

[![License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2502.XXXXX-red.svg)](https://arxiv.org/)

---

## 📖 Overview
**SanskritRasaBank** is a state-of-the-art dataset and benchmarking suite designed to bridge the gap between classical Indian aesthetics (**Nava-Rasa**) and modern Natural Language Processing. 

For over two millennia, the *Nāṭyaśāstra* framework has defined nine fundamental "essences" of human emotion. This project provides the first computational grounding for these rasas at scale, featuring **23,028 human-validated annotations** from the *Vālmīki Rāmāyaṇa* and other classical texts.

### The Nine Rasas (Nava-Rasa)
| Rasa | Meaning | Dominant Emotion |
|:---:|---|---|
| **Śṛṅgāra** | Love / Beauty | Rati (Love) |
| **Hāsya** | Laughter / Humor | Hāsa (Mirth) |
| **Karuṇā** | Grief / Compassion | Śoka (Sorrow) |
| **Raudra** | Fury / Anger | Krodha (Anger) |
| **Vīra** | Heroism / Valor | Utsāha (Enthusiasm) |
| **Bhayānaka** | Terror / Fear | Bhaya (Fear) |
| **Bībhatsa** | Disgust / Revulsion | Jugupsā (Disgust) |
| **Adbhuta** | Wonder / Amazement | Vismaya (Wonder) |
| **Śānta** | Serenity / Peace | Śama (Calmness) |

---

##  Dataset Statistics
Our corpus was constructed using a **validated LLM-ensemble framework** (GPT-4o, DeepSeek-V2, LLaMA-3.1) and audited by an expert Sanskrit Philologist (94% agreement).

*   **Total Samples (High-Confidence):** 12,548
*   **Total Corpus (Global Pool):** 29,902
*   **Validation:** Expert-checked 500-sample probe.
*   **Balancing:** Augmented the primary *Rāmāyaṇa* corpus with supplementary texts to fix the 'rare-rasa' (Hāsya, Bībhatsa) tail.

---

##  Model Benchmarks
We fine-tuned across four major multilingual architectures. **MuRIL-large** emerged as the state-of-the-art specialist for Sanskrit Rasa classification.

| Model | Accuracy (%) | Weighted F1 | Macro F1 |
|:---|:---:|:---:|:---:|
| **MuRIL (Specialist)** ⭐ | **88.99** | **89.05** | **90.42** |
| IndicBERT-v2 | 86.71 | 86.83 | 87.29 |
| XLM-RoBERTa-large | 84.52 | 84.61 | 85.03 |
| SanBERTa | 61.40 | 61.53 | 59.87 |

> **Efficiency Win:** Our 236M-parameter MuRIL matches GPT-4o performance on overall accuracy while beating it on **Macro-F1 (+1.26 points)**, demonstrating the power of domain-specific fine-tuning.

---

##  Project Structure
```bash
SanskritRasaBank/
├── data/               # Raw, Annotated, and Filtered (Gold) datasets
├── models/             # fine-tuning scripts for MuRIL, IndicBERT, etc.
├── annotation/         # The 3-LLM Ensemble Pipeline source code
├── evaluation/         # Literary analysis & error tracking (Sankey/Radar)
├── results/            # Pre-computed metrics, confusion matrices, and predictions
└── notebooks/          # Exploratory analysis and literary visualizations
```

---

##  Quick Start

### Installation
```bash
git clone https://github.com/suhritghimire/NavaRasaBank.git
cd NavaRasaBank
pip install -r requirements.txt
```

### Inference Snippet
```python
from transformers import pipeline

classifier = pipeline("text-classification", model="suhritghimire/navarasa-bank")
verse = "तो रामो महातेजा भ्रातरं निहतं रणे। दृष्ट्वा क्रोधसमाविष्टो बभूव ज्वलिताननः॥"
print(classifier(verse)) 
# Output: [{'label': 'Raudra', 'score': 0.941}]
```

---

##  Citation
If you use this project or dataset, please cite our ACL paper:

```bibtex
@inproceedings{ghimire2025navarasabank,
  title={Nine Flavors: NavaRasaBank, an LLM-Ensemble Corpus for Computational NavaRasa Analysis},
  author={Ghimire, Suhrit and {et al.}},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  year={2025},
  publisher={Association for Computational Linguistics}
}
```

---

##  Acknowledgments
Special thanks to **Prof. Rohini Raj Timilsina** (Tribhuvan University) for the expert validation of the corpus labels. 

---
© 2025 NavaRasaBank Team. Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) & [MIT](LICENSE).
