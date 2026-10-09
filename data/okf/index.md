---
type: Index
title: "MedRAG_OKF Medical Knowledge Bundle"
description: "OKF bundle of patient-education concepts used by the MedRAG_OKF chatbot."
---

# MedRAG_OKF medical knowledge bundle

An [Open Knowledge Format](https://cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing) bundle: one concept per file, the file path is the concept identity, concepts link to each other with markdown links.
All concepts are currently `status: draft` AI-written prototype summaries; a human reviewer marks a concept as verified with `python cli.py okf-verify <path> --by <name>`.

- [Conditions and symptoms](/conditions/index.md) (11 concepts)
- [Infections](/infections/index.md) (3 concepts)
- [Emergency guidance](/emergency/index.md) (1 concepts)
