# HyperTopo-Adapters: Mixed-Curvature Latents + Topology-Preserving Losses

This repo implements a product-manifold latent head (H⊕E⊕S) atop a frozen tokenizer, with hyperbolic metric learning and topology-preserving regularization (persistent homology) for micro-lesion segmentation.

Highlights:
- Strict **Euclidean-only** baseline (`model.euclidean_only: true`)
- **Qualitative dump** utility (`scripts/dump_quali.py`)
- **Ablation sweep** + **collation to CSV & LaTeX table** (`scripts/collate_results.py`)

See this README and `paper/checklist.md` for a smooth path to a workshop paper.
