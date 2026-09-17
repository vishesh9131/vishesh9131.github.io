---
layout: post
title: LLMs Count Digits, Not Store a Number Line
date: 2026-06-11
description: Constraint tests on how language models represent numerical magnitude.
tags: [LLM, representation learning, latent space]
categories: [research]
---

I got stuck on a dumb sounding question: when a language model sees `4521`, what is actually in the vector?

People say magnitude is linearly decodable. Ok fine. But does the model keep a real quantity like a number line? Or just rank order? Or digit symbols glued together? Everyone cites probes, nobody lines the stories up on the same stimuli.

So I spent a long stretch building constraint tests across a bunch of models (small Qwen up to DeepSeek 70B, Llama, Phi, GPT-2, BERT, etc). Not to confirm my favorite theory. Mostly to break theories.

This is my informal writeup of that work. Numbers live in `Experiment2/results/*_verdict.json` if you want the raw logs.

## the trap that fooled me early

If you sample integers uniformly from 1 to a million, value, log value, and rank all move together. Monotone.

So a probe that only cares about ordering cant tell you if the model encoded cardinal size, ordinal rank, or digit pattern. I formalized that in the writeup as a non-identifiability thing. Practical fix: use **gapped** number sets where rank and log value disagree, plus cross val, shuffle nulls, bootstrap CIs. Dont trust single split R² from a uniform grid.

Also intrinsic dimension on 100 points in 1500d looks "thin" for fake reasons. Small sample. I had to stop trusting my own earlier manifold plots until I reran with controls.

## what I was trying to kill

Six rough accounts from the literature:

1. Cardinal / linear value code
2. Pure ordinal (only order matters)
3. Symbolic digit code
4. Log compressed magnitude manifold
5. String + context entangled hybrid
6. CCM (my synthesis name at the end)

I wrote down 16 constraints any story has to survive. Things like: log decodable, no vector arithmetic, digit identity readable, magnitude weak vs total variance, no shared axis across domains, causal necessity, notation robustness, etc.

The scorecard table in my notes is just a map. Experiments carry the weight, not the tick count.

## headline results (the ones that survived)

### magnitude is real but squashed

Log value probes work across models and domains. R² often 0.94 to 1.00 on the setups I ran.

But geometry is **compressed**. Between/within cluster ratios κ around 2 to 4.8. Isometry to log distance is moderate (ρ ~ 0.2 to 0.5). So yes metric-ish in log space, not a clean ruler.

Partial distance tests on gapped sets: log value still beats pure rank in the large encoders (bootstrap CI excludes 0 for Llama3 / DeepSeek on several domains). Small effects on Qwen 7B integer and GPT-2 distance were marginal. I report those without pretending they're huge.

### order of magnitude ≈ digit count (length), read positionally

This was the mechanism result that stuck.

Decompose log10(n):

- floor part (# digits / order of magnitude) probes insanely high (R² ~0.976 to 0.982 on Qwen/Llama/DeepSeek)
- fractional part / leading digit adds correction
- last digit depends on readout (pooling)

So the model is mostly **counting digits in the string**, not holding an abstract scalar. Log compression falls out of positional notation.

Notation test backed this: same value as plain digits, comma form, scientific, words. Log still decodable 0.90 to 0.99.

Zero pad to fixed width (kills raw token length cue): R² drop ≤0.04, mean ~0.015. Magnitude still there. So it's **where significant digits sit**, not raw token count per se.

### no shared number line across contexts

Same numeric value in different templates (population vs distance vs price vs weight, eight domains total).

Probe **directions** for magnitude barely align. Cross domain cosine mean ~0.12 to 0.25. Above random noise in high D, far below "same axis."

Pooled leave-one-domain-out R² weak (up to ~0.46 on Qwen, negative on Llama3 in one setup). So you dont get one universal magnitude direction.

But magnitude **information** still transfers somewhat (calibration free Spearman 0.53 to 0.97 depending model/domain). My read: context picks a new readout direction each time, order info is partly portable, not a shared register.

### no vector space arithmetic

Addition composition in embedding space doesnt work like vector add. Product from concat embeddings fails OOD. Group symmetries absent (that constraint is noisy but directionally right).

So dont expect h(347)+h(892) ≈ h(1239).

### causal erasure (the part that isnt just probes)

On Qwen2.5-1.5B comparison task ("is a > b?"):

| condition                           | accuracy      |
| ----------------------------------- | ------------- |
| baseline                            | 0.91          |
| erase magnitude subspace all layers | 0.53 (chance) |
| erase random rank-8 subspace        | 0.67 ± 0.15   |

Erase **length** (#digits) subspace: same-order pairs mostly ok (0.90), cross-order hurt (0.62).\
Erase **leading digit** subspace: same-order collapses (~0.50).

Magnitude subspace and length subspace overlap heavily (cos ~0.89). Leading digit more distinct (cos ~0.43). Double dissociation at 1.5B.

Replicated magnitude collapse at 7B (0.55 vs baseline ~0.92). Length-only selective effect was cleaner at 1.5B; at 7B length erasure looked more redundant. I dont oversell that as universal.

Point: the comparison behavior **needs** the magnitude/length directions. Geometry isnt decorative.

### steering experiment failed (reporting it anyway)

I tried adding one domain's magnitude direction into another domain's residual stream mid layers. Hoped for weak cross domain shift.

Within domain moved judgments. Cross domain looked smaller. Then random direction control moved things similarly. After correction, contrast unreliable. Generic perturbation bias. I dont use steering as evidence. Correlational orthogonality still stands.

## the synthesis name: CCM

**Contextual Compositional Magnitude** is just my label for what didnt die:

- Numbers are built from **notation tokens** (digits etc)
- A weak log compressed magnitude summary is linearly readable on top
- Dominated by **order of magnitude / digit count**, with leading digit correction
- Encoded along **context specific directions** (no shared axis)
- Digit identity coexists
- No vector arithmetic
- Causal erasure supports length/magnitude link for comparisons

CCM isnt claiming I discovered log compression or probing. Lot of that exists in 2025 papers. My bit is adjudication + identifiability caution + length mechanism with erasure + no-shared-direction result.

On the internal scorecard CCM marks all 16 constraints. Closest competitor is the string/context hybrid account at 13. Take the table as organization, not proof.

## stuff I am not saying

- Models have a human like number line in latent space
- One global magnitude neuron direction across all prompts
- Probes prove how arithmetic is implemented during reasoning
- CCM is final cognitive science of numeracy
- Steering negative means context doesnt matter (it does, correlationally)

## methods compressed

Integer templates, eight ordered domains, gapped geometries for identifiability, repeated CV probes, bootstrap where noted, twelve models total (nine core + three for notation battery), causal all-layer subspace projection on Qwen 1.5B and 7B.

Code under `Experiment2/` experiments. Key json: `ablation_verdict.json`, `notation_verdict.json`, `domains_verdict.json`, `transfer_refine.json`, `bootstrap_ci.json`, `geometry_verdict.json`, `steering_verdict.json`.
