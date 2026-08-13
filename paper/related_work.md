# Related Work

## Literature Table

### Data Selection for Instruction Tuning

| Paper | Method | Dataset | Main Result | Limitation (relevant to this work) |
|---|---|---|---|---|
| **DEITA** (Liu et al., ICLR 2024) | Scores instructions on complexity + quality (via trained scorer models) + diversity (embedding similarity threshold); selects smallest effective subset | 300k pool → 6k selected, LLaMA/Mistral 7B–13B | Achieves results comparable to all 300k samples using only 3k — 100× reduction; selection outperforms random subsets consistently | Selects a subset but does not test training *order*; uses LLM-based scorers at 7B–13B scale, not lightweight scorers at sub-1B PEFT scale |
| Data Selection for LLM Instruction Tuning (survey, arxiv 2402.05123) | Surveys quality scoring, diversity/coverage, and influence-based selection methods | Cross-paper | Frames DEITA as integrating complexity, quality, and diversity — complexity capturing instruction length/difficulty, quality capturing output accuracy | Survey; no new experiments. Use for framing the field, not as a comparison baseline |

### Curriculum Learning for LLMs

| Paper | Method | Dataset | Main Result | Limitation (relevant to this work) |
|---|---|---|---|---|
| **DUCL** (Jiang et al., AAAI 2026) | Argues pure difficulty-based curriculum is insufficient; combines difficulty with a utility signal (each sample's actual contribution to performance improvement) | Fine-tuning benchmarks | Difficulty alone is not enough — utility-weighting is needed for robust curriculum gains | Direct caveat: our curriculum signal is pure complexity, not utility-weighted. Preempt in Limitations as future work |
| **EDCO** (Pang et al., ICML 2026) | Dynamic curriculum that adapts during training rather than using a fixed pre-training order | Domain-specific fine-tuning | Static curricula lack adaptability to the model's evolving needs; dynamic ordering outperforms | Frames our easy→hard approach as the natural "static curriculum" baseline case — cite when positioning our method |
| Human-Inspired Learning Strategies in Medical QA (arxiv 2408.07888) | Tests blocked vs. interleaved vs. standard curriculum across multiple LLMs | Medical QA | LLM-defined difficulty measures produced the largest gains; gains are domain-specific and often modest | Domain-specific; supports "null or modest result is legitimate" framing for our Discussion |

### Parameter-Efficient Fine-Tuning

| Paper | Method | Dataset | Main Result | Limitation (relevant to this work) |
|---|---|---|---|---|
| **LoRA** (Hu et al., NeurIPS 2021) | Freezes pretrained weights; injects trainable low-rank matrices (ΔW = BA) into attention layers; matrices merge at inference, no latency overhead | GLUE, various LLMs | Reduces trainable parameters by orders of magnitude while matching full fine-tuning performance | Original paper predates instruction-tuning-at-scale; we extend it into a selection + curriculum context it was not designed for |

---

## Gap Statement

*(Write once here. Copy verbatim into Related Work section of the paper. Do not rewrite.)*

> DEITA and related work establish that complexity- and diversity-based data selection improves instruction-tuning efficiency, achieving near-full-dataset performance with as little as 1–6% of the data. However, this line of work does not examine whether the *order* in which selected examples are presented to the model adds further benefit beyond selection alone. Separately, the curriculum learning literature (DUCL, EDCO) explores difficulty-based ordering for LLMs but does not combine this with principled data selection, and does not evaluate at sub-1B PEFT scale. **We address precisely this gap:** using a lightweight, heuristic-based importance scorer (no external LLM required), we isolate the independent and combined effects of selection and curriculum ordering in a controlled 2×2 experiment at sub-1B LoRA scale on the Alpaca dataset.

---

## Notes for Paper Writing

**DEITA row — what to say in Related Work (verbatim):**
> Liu et al. (2024) introduce DEITA, which scores instruction data on complexity, quality, and diversity using trained scorer models, then selects the smallest effective subset. DEITA achieves results comparable to training on 300k samples using only 3k — a 100× reduction — demonstrating that data quality dominates quantity for instruction tuning. However, DEITA selects a subset but does not test whether the training *order* of that subset adds further benefit. Additionally, DEITA uses 7B–13B LLM-based scorers, which are not feasible at sub-1B PEFT scale. Our scoring function is DEITA-inspired but uses lightweight heuristics (embedding distance + length proxies), and our primary question is whether ordering the selected subset by difficulty adds anything beyond selection alone.

**DUCL caveat — what to say in Limitations (verbatim):**
> Our curriculum signal is pure instruction complexity (length + vocabulary richness), not utility-weighted. Jiang et al. (2026) argue that difficulty alone is insufficient and that weighting examples by their actual contribution to model improvement (utility) yields more robust curriculum gains. Incorporating a utility signal is a direct avenue for future work.

**Framing if curriculum result is null or weak:**
> Selection improves data efficiency consistent with DEITA; curriculum ordering shows [no / marginal] additional benefit at this model scale and dataset. The DUCL and Medical QA results suggest that curriculum gains are sensitive to the difficulty signal used and the domain — both factors differ from the settings where curriculum learning has shown the strongest effects.
