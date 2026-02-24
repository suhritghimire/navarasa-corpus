# 🏛️ NavaRasa Corpus

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dataset Size](https://img.shields.io/badge/Verses-17%2C147-green)]()
[![Annotators](https://img.shields.io/badge/Annotators-4%20LLMs-orange)]()

**NavaRasa Corpus** is the first large-scale Sanskrit verse dataset annotated with *Navarasa* — the nine aesthetic sentiments (*rasas*) from Indian classical aesthetics theory (Nāṭyaśāstra of Bharata Muni).

> 📄 **Paper**: *NavaRasaBank: A Multi-Annotator Sanskrit Rasa Classification Corpus* — under review at ACL 2025.

---

## 🎯 Overview

Sanskrit literature is deeply governed by *rasa* theory. This dataset enables computational study of aesthetic sentiment in classical Sanskrit texts for the first time at scale.

| Metric | Value |
|--------|-------|
| Total Verses | **17,147** |
| Annotators | **4** (GPT-4o-mini, DeepSeek-chat, Llama-3.3-70B, Gemini Flash) |
| Sources | 12+ GRETIL texts |
| Languages | Sanskrit (Devanāgarī) |
| Label Format | Single rasa per verse |

---

## 🌸 The Nine Rasas

| Rasa | Meaning | Bhāva |
|------|---------|-------|
| Śṛṅgāra | Love / Romance | Rati |
| Hāsya | Humor | Hāsa |
| Karuṇa | Compassion | Śoka |
| Raudra | Fury | Krodha |
| Vīra | Heroism | Utsāha |
| Bhayānaka | Terror | Bhaya |
| Bībhatsa | Disgust | Jugupsā |
| Adbhuta | Wonder | Vismaya |
| Śānta | Serenity | Sama |

---

## 📚 Sources

Drawn from 12+ classical Sanskrit texts via [GRETIL](http://gretil.sub.uni-goettingen.de/):
- Abhijñānaśākuntalam (Kālidāsa)
- Meghadūta (Kālidāsa)
- Raghuvaṃśa (Kālidāsa)
- Subhāṣitaratnakoṣa
- Mahāsubhāṣitasaṅgraha
- Kathāsaritsāgara
- Kirātārjunīya
- Bhallaṭaśataka
- Harikeli-Nāṭaka Prīti
- Harirdatta-Ratna
- And more...

---

## 🏗️ Pipeline

```
build_gretil_dataset.py       # Scrape & extract verses from GRETIL HTML
annotation/pipeline.py         # Multi-annotator ensemble labeling
label_final_openai.py          # GPT-4o-mini (OpenAI Batch API)
label_final_deepseek.py        # DeepSeek-chat (parallel)
label_groq_70b.py              # Llama-3.3-70B via Groq
openai_batch_retrieve_all.py   # Merge batch results
```

---

## 📊 Agreement Statistics

| Pair | Agreement |
|------|-----------|
| OpenAI vs DeepSeek | **57.0%** |
| Groq vs OpenAI | 37.1% |
| Triple Agreement | **27.1%** |

**High-confidence gold set** (2+ of 3 annotators agree): **8,726 verses**

---

## 🚀 Quickstart

```bash
pip install pandas openpyxl
python build_gretil_dataset.py
```

---

## 📋 Citation

```bibtex
@dataset{ghimire2025navarasa,
  title     = {NavaRasa Corpus: A Multi-Annotator Sanskrit Rasa Classification Dataset},
  author    = {Ghimire, Suhrit},
  year      = {2025},
  note      = {Under review at ACL 2025}
}
```
