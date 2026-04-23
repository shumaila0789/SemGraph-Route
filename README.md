# SemGraph-Route

Official implementation of **SemGraph-Route: Zero-Shot Semantic Topological Graph Routing via Space Type Priors for GPS-Denied UAV Navigation**.

---

## Requirements

- Python 3.12
- `transformers==4.40.2`
- See `requirements.txt` for full dependencies

```bash
pip install -r requirements.txt
```

---

## Data

Download the following datasets and place them under `data/`:

- **TartanAir:** https://theairlab.org/tartanair-dataset/
  - Environments used: `abandonedfactory` (train), `hospital` (test)
- **EuRoC MAV:** https://rpg.ifi.uzh.ch/docs/IJRR17_Burri.pdf
  - Sequences used: `MH_01_easy`, `V2_01_easy`

For EuRoC, place zip files in `data/euroc/` and run:
```bash
python download_and_label_euroc.py
```

---

## Run Order

```bash
# 1. Space-type classification
python phase1_space_labeling.py --env abandonedfactory
python phase1_space_labeling.py --env hospital --limit 100

# 2. Prior learning
python learn_space_priors.py

# 3. Semantic scene graph construction
python build_scene_graph.py

# 4. Main evaluation (all baselines + bottleneck)
python phase3_semgraph_planner.py

# 5. Ablation study
python ablation_space_types.py

# 6. Zero-shot generalization
python validate_prior_stability.py

# 7. Hospital evaluation
python evaluate_hospital.py

# 8. Additional metrics and figures
python finalize_experiments.py
```

All results and figures are saved to `data/semgraph_results/` and `figures/`.

---

## VLM Note

Use MoondreamV2 via HuggingFace Transformers — do **not** use the `moondream` pip package.

```python
AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-01-09",
    trust_remote_code=True,
)
```

---

## Citation

```bibtex
@article{javaid2025semgraph,
  title={{SemGraph-Route}: Zero-Shot Semantic Topological Graph Routing
         via Space Type Priors for {GPS}-Denied {UAV} Navigation},
  author={Javaid, Shumaila and He, Bin and Saeed, Nasir},
 }
```

---

## License

MIT License
