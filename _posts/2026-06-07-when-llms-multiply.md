---
layout: post
title: When LLMs Multiply
date: 2026-06-07
description: Right ballpark, wrong answer - and what hidden states reveal.
tags: [LLM, numerical reasoning]
categories: [research]
---

When LLMs Multiply: Right Ballpark, Wrong Answer

I was running multiply prompts on our local GPUs one evening. Same template every time, greedy decode, no chain of thought. And I kept getting annoyed at the answers.

The model sounds sure of itself. The number looks fine at a glance. About the right length. Like someone who knows the answer is "big" but never did the multiplication.

So I started saving outputs properly. Then I wondered what the internal state looks like right before it prints anything. That turned into a bunch of probe runs on the same box.

Writing this down for myself mostly. If you read the verdict jsons later, this is the english version.

## Part 1: outputs looked wrong in a specific way

### what I ran

Prompt:

> Question: What is 347 times 892? Give only the number.\
> Answer:

No CoT. Greedy. Max 12 tokens. Mostly 2 to 3 digit operands. Few hundred pairs per model.

Two scores:

- **Exact**: is the number actually correct
- **OOM** (order of magnitude): right ballpark, same digit count basically (same floor of log10)

### numbers

Exact multiply is bad. Scale is oddly good.

| Model        | Exact | OOM (right scale) |
| ------------ | ----- | ----------------- |
| Qwen2.5-1.5B | 7%    | 93%               |
| Qwen2.5-7B   | 22%   | 100%              |
| Llama3-8B    | 12%   | 100%              |
| Phi-3-mini   | 13%   | 74%               |
| Mistral-7B   | 3%    | 79%               |

Qwen 1.5B: 93% get the scale right. 7% get the product.

I call it **Fermi decoding** bc its like a Fermi guess. Right size, wrong value.

### sanity checks I ran bc first pass felt too weird

**Token limit?**\
Bumped max_new_tokens 12 to 48. Exact still ~9%. So not truncation.

**All ops broken?**\
Same prompt style for **addition** on Qwen 1.5B: 92% exact, 99% OOM. Add is fine. This is mostly a multiply thing with this template.

**Wrong answers shape**\
On 1.5B wrong multiply answers, 90.6% still OOM-correct. Usually right length, wrong digits. Not random garbage.

**Scale in hidden state before decode?**\
Probed last prompt token. Linear probe on product scale R² ~0.82 to 0.85, about same as digit count probe. So scale info is often already there internally. Feels like the model fails at spitting out exact digits, not at having zero idea of size.

If your benchmark only checks "did it output a plausible sized number" you will overrate these models on multiply.

## Part 2: probing hidden states

After logging outputs I wanted to know if multiply vs add look different inside when operands are the same.

Linear probes on hidden states. Ridge regression, cross val, shuffle nulls (shuffle labels, probe should collapse). Mainly Qwen2.5-1.5B, some reruns on 7B and Phi-3.

### times vs plus is decodable, but vectors are almost the same

Binary probe for operation word: R² ~1.0 on three models. Shuffle nulls negative.

Then cosine between mul and add last-token states for same (a,b): ~0.9999. Basically the same vector.

So yeah you can read off times vs plus. But geometrically its not two separate clusters. More like a tiny shift on top of almost identical states.

### log product readable in add prompts too (boring reason)

Layer 4:

- mul prompt log10(a\*b): R² ~0.987
- add prompt log10(a\*b): R² ~0.985
- gap 0.002

Both prompts still have a and b in the text. Probe can pull operands and combine. Dont read that as "add mode stores multiplication."

### operand tokens carry most of it

Operand digit hiddens -> log10(product) R² ~0.98. Last token only adds ~0.01 on top.

Bilinear h_a \* h_b does worse than concat. So not some clever multiplicative geometry, mostly linear readout.

Static embeddings of operands predict mul prompt state R² ~0.73. Context changes the vector. Magnitude still easy to read once youre in the prompt.

### digit tokens

Digit identity R² ~0.97, place R² ~0.95 on operand digit tokens in mul prompts.

But token index on same rows: R² ~0.99. So "place" might just be where you are in the sequence. I wouldnt claim abstract place value from this alone.

### templates matter a lot

Static Integer:n vs mul prompt cosine ~0.32.

Static vs dumb filler prompt ("summarize weather in Paris") cosine ~0.21. So low cosine is partly just different prompts, not magic arithmetic geometry.

Reword mul template, same numbers: cosine ~0.72. Wording changes the state.

### nudging add toward mul doesnt flip magnitude probes

Shift add state along mean(mul - add). Classifies op fine. log10 product vs log10 sum probe scores basically dont move. Direction tags the op, doesnt act like a magnitude switch.

### probes vs actual output

Qwen 1.5B layer 2: log10(product) probe R² ~0.996\
Same setup greedy multiply: 7% exact, 93% OOM

Hidden state looks informative. Output often wrong anyway. Decodable doesnt mean the model uses it correctly at decode time.

## putting the two halves together

|                | output logging      | probes                          |
| -------------- | ------------------- | ------------------------------- |
| exact multiply | low on Qwen         | didnt focus on this             |
| scale / OOM    | high on Qwen        | log product R² ~0.99 in prompt  |
| where it fails | generation          | not early scale readout         |
| mul vs add     | Fermi mostly on mul | op readable, states nearly same |
| causal         | no                  | no, correlational               |

How I explain it to someone in the lab:

1. Before tokens come out, scale is often already linearly readable.
2. When tokens come out, scale is right a lot, exact digits wrong (Fermi thing).
3. Mul and add prompts for same numbers are almost the same vector; op is a small extra bit.
4. Magnitude probes dont separate ops cleanly. Operands in the prompt dominate.
5. R² 0.99 on log product is not "solved multiplication." The 7% exact rate already said that.

## stuff I am not saying

- found an arithmetic circuit inside the transformer
- mul and add live in different geometric worlds
- high product R² in add prompts means its doing mul secretly
- place probes prove positional notation (token index confound is too strong)
- decodable = used = causal

## methods in one breath

Fixed templates. Forward pass for probes, greedy for outputs. Usually last prompt token. Ridge, 5-fold CV repeated, shuffle nulls. Numbers in `cursor-work/results/`, code in `cursor-work/experiments/`.

## if you benchmark this stuff

Split exact and OOM on multiply. Dont use probe R² as a competence score. If you care about operation understanding use filler prompts and paraphrases and position baselines. Add under same terse format is much easier, dont assume all ops look like mul.

Started bc multiply answers looked like good guesses with wrong exact digits. Still does across the models I tried. Inside, scale probes easy, op word decodable but tiny on shared state, magnitude doesnt split clean by operation.

Gap between probe R² ~0.99 and 7% exact is what stuck with me. Feels like readout/generation is where it breaks, not that magnitude is totally absent from representations.

Wouldve helped me to read something like this before prompt 200.

Vishesh
