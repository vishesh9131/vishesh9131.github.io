---
layout: post
title: Tokenizers Are All We Need
date: 2025-11-07
description: A grounded walkthrough of tokenization and its impact on model behavior.
tags: [tokenization, LLM, Mamba]
categories: [research]
featured: true
---

# Sitting Outside IISc, Freezing a Little, Thinking About Tokenizers

I wrapped up my work and i am sitting in front of the main building of the Indian institute of science…its so soothing cold the winds of november are giving my mind thrills…

Ok so we are about to talk about tokenizers today!!!
Why does it come to mind??? As we go through reading foundational models …"How LLMs understand english is one of the many questions readers come around!"

And tokenizer is what breaks sentences into tiny tokens, tokens are not words, not characters not syllables, tokens are subword units chosen to make the entire text space efficient to represent. The cleverness and strategy while tokenizing the corpus we take decides how well our model is able to grasp the context and generate the next token with a good tpc (tokens per character)(lower is better) ohhhh what's tpc now!!!! …calm down we will talk about this now, it's just a simple measure of how much total token your model can generate divided by the total number of characters in the text…

For example : if the gpt-pro-max can generate 11 tokens for the sentence "Who is Ramanujan??" and on the other side gpt-potato generates 23 tokens for the same sentence..can you spot the time and space complexity which version of gpt will take more??? Yes you got it, that's gpt-potato… because to represent a entity he is taking more chars then gpt-pro-max….

Yeah moving forward while keeping the tone of discussion light we will talk and in between i will keep making you know the further concepts tokenizers are associated with or famous tokenizers how they cleverly make text space efficient.

You must be wondering how they look, is there any pattern or symmetry between them…

If you are not thinking this, okay be calm i will make you think. You see every tokenizer when get trained it makes a finite list of tokens that finite list of tokens we called vocabulary and using that we make every words which is in the text space so just like 26 alphabet is vocab of english lang…Tokenizers invent an alphabet, but the letters are allowed to be as big or as weird as they want, as long as the final reconstruction cost stays small.

According to some notable papers , they reveal promising benchmarks on upgrading and working on optimization of tokenizers as we know earlier researchers were only interested in thinking about optimizing the architecture or inventing one…
This is how impactful tokenizers can be!!!!!

---

## Encode Decode Logic

Tokenizers are bound by a deterministic rule which is very senseful, just like when you translate a sentence in hindi , it should not get changed when you translate it back to english. Similarly when we tokenize the corpus of text we expect the same text corpus if we """detokenize"""" it in future. detokenize wired words right!!! that's why we introduce encode and decode nomenclature … when you tokenize something you are encoding it and decoding it when you need your corpus back to its original state.

Great now you become a geek who knows basis of toeknizers…lets take this article further and i will discuss you about some of the coolest topics, my friends who 1 year back sit on a placement they were telling me interviewers are really interested if one knows about modules which makes generative models work …everyone knows how gpt works right transformers transformers transformers but knowing what qkv scoring is , knowing how multi head attention is being computed or knowing the working of positional encoding is what keeps your answers and replies in placement healthy…

Just like i told you about tpc…there are so many entities to judge these tokenizers we call it Tokenization cost metrics….you are sitting in a MAANG interview for gen ai JD and interviewer is not asking TCM….thats where your dream will break and you are not really giving an interview…lol

---

## TPC, TPW, Entropy, OOV

Ok back to the topic! I told you how tpc can judge which tokenizer is efficient and cost effective similarly we are blessed with more TCMs like: Token Per Word (TPW), Tokenization Entropy, OOV Rate (Out-of-Vocabulary)...

We will discuss one by one ; just like TPC tells us how expensive a tokenizer is for a language…TPW helps readers gauge efficiency quickly but according to me it's less scientific. "WHYWHY VISHESH???" I dont know!!!! It just doesn't feel scientific to me.

I am so unpredictable lol..just like me there is a term to judge how unpredictable the token boundaries are for that language and tokenizer that is called Tokenization Entropy. And for domain specific writing like medicine or cook book to make veg food, How often tokenizers fail to represent a term and have to back off to characters is called its OOV Rate aks Out-of-Vocabulary Rate.

---

## Some Under-the-hood Ideas

Now there are some more concepts which do matter in terms of tokenizers but works under the hood , i have written their definition you can read if you like:

- **Sub-word Segmentation** : Every modern tokenizer relies on the idea..Instead of whole words, break text into the most useful word-pieces.
- **Merge Algorithms** : How tokenizers decide which word-pieces deserve to become tokens.
- **Frequency-driven Compression** : All common tokenizers are compression schemes pretending to be linguistic.

---

## Main Tokenizer Families

NOW NOW NOW We will talk …oh who switched my speaker off! Ok …. NOW NOW NOW WE WILL TALK ABOUT THE MAIN TOKENIZER FAMILIES AND THEIR WORKING..huh that was lowd.

One of the earliest used tokenizers was Byte-Pair Encoding(BPE) used in GPT-2/3, BLOOM…and working on these will cost me a whole new article so I will give you an intuition!

---

## BPE Intuition

BPE starts with characters -> counts which pair of characters appear together the most -> merge those pairs and treat them as a single symbol(merge tables) -> repeat 100000+ times ->Final merges -> become the vocabulary.

Merge table is an ordered list of pair-merge decisions the tokenizer learned during training.

It's simple, deterministic, and compresses well, but it struggles with languages where words morph heavily.

It looks like this:

```
0: ("e", "r") → "er"
1: ("pl", "ay") → "play"
2: ("play", "er") → "player"
3: ("player", "s") → "players"
```

### Example of how Encodings uses the Merge Table in BPE:

**Start text** = token

**Merge table it formes:**

```
1: ("t", "o") → "to"
7: ("k", "e") → "ke"
10: ("to", "k") → "tok"
52: ("ke", "n") → "ken"
100: ("tok", "en") → "token"
```

**possible_doubt** : why not ("e","n") -> "en"

Because merging a pair in a merge table is not logical! BPE never decides merges based on logic.It only follows the statistics of the training corpus.

**Encodings happens like:**

- First look for the earliest merge number that applies.
- Merge that pair
- Keep going
- Eventually, it might end up with the full token "token" if enough rules exist.

---

## SentencePiece / Unigram LM

Google has a famous library called Sentence piece where they have introduced Unigram Language Model Tokenizer, using which all the LLaMA family is trained on!!!

So unlike BPE instead of merging it starts with a giant list of candidate sub-words and trains a probabilistic model. The goal of the model "what is the most likely set of subwords that explains all sentences???" and iteratively prunes the vocabulary.

It handles multilingual and weird morphology better than BPE.
It generates multiple possible tokenizations and picks the best.

But google didn't come up with sentence pieces in the first place Early google models like BERT, DistilBERT were trained on different tokenizers known as WordPiece.
Its where Google was transitioning their aura form BPE to Unigram LM Tokenizer.

So that transition development is what they call WordPiece. It's very similar to BPE but with different scoring objectives, they asked "if I add this subword to vocabulary will it help me to explain my text better?????" and thats all maximize likelihood really meant.

WordPiece aften ads "##" markers to show subword continuations. People still talk about this because it introduced the idea that tokenization can be tightly coupled with the LM's training data.

---

## Byte-Level BPE

I told you BPE was used in GPT-2 / 3…OpenAI really took tokenizers seriously and while hunting they found Byte-Level BPE they have used in GPT4 and Mistral also comes into account for using Byte-Level BPE.

How does it work ?? same as bpe, but it starts from raw bytes(0-255) instead of characters.that means every piece of text is representable thus no OOVs.
It became dominant because now it can handle emojis, accents, programming languages, and rare symbols effortlessly.

---

## TikToken / GPT-4 Encodings

But now OpenAI uses there own tokenizers they call it tiktoken / GPT-4's Encodings
As it introduces modern pretokenization rules, special tokens, optimized merges, extreme speed by memory mapped tables; i will talk about this now …

People says tiktoken is nothing but "BPE but fast" that's very surface level…because underneath, there are a few design moves that make these tokenizers behave cleaner, tighter, and more predictable than classical BPE.

---

### Pretokenization Rules

Pretokenization Rules ie OpenAI tokenizers don't treat raw text as-is. They apply collapsing weird whitespaces patterns into std ones, normalizing some unicode edge cases, treating punctuation in more consistent way, spitting boundaries around certain characters…its not like full normalization (like NFKC) it's more like "Fix the obvious inconsistencies but don't destroy meaning."

---

### Special Tokens

Special Tokens : Gpt model rely on a fairly rich set of reserved tokens i.e. end-of-text, beginning-of-assistant message, beginning-of-user-message, system prompts, document seperators, bye-level fallbacks and some metadata tokens for chat formatting… It's not at all impressive right, but together they let the model understand where a conversation turn begins, what role a segment plays, and how to differentiate data from instructions.

Older tokenizers basically ignored structure, tiktoken has just baked structure into vocabulary itself

---

### Optimized Merge Table

Optimized Merge Table : You know BPE generates very large, redundant, slow scan merge tables.OpenAI reworks this table into something extremely compact and efficiently indexable.Every merge is packed in a way that reduces cache misses and speeds up lookup time during encoding.

---

### Memory-Mapped Tables

Memory-Mapped Tables for Speed : I can say this is pure system engineering,
Instead of loading merge rules into RAM and parsing them at runtime, tiktokenn uses memory mapped files: the OS loads the pages you actually touch, random access is extremely fast, zero copy overhead.

---

### Soft Normalization

Soft Normalization : This is the part most people are not aware of. Older BPE tokenizers got wrecked by slight Unicode variations: smart quotes, accent marks, zero-width spaces, exotic whitespace… All these caused fragmentation, meaning the tokenizer produced way too many tokens for the same concept.
