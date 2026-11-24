# P18 (self-verification for open-vocabulary MER): novelty log, round 1 (2026-11-24)

## Claim and scoop scenarios

**Core claim (to be tested):** a training-free decoder for generative MER that
- scores every candidate emotion label from the model's own samples by the same MLLM's yes/no judgement on the clip (probability of "Yes"); and
- fuses that probability with the sample consensus inside an emotion-wheel expected-F set decoder (CONSTRUCT).

**Scoop scenarios:**
- (a) an MER / affective paper verifies candidate emotion labels with an MLLM yes/no question at inference;
- (b) generate-then-verify, or re-ranking, of open-vocabulary emotion labels;
- (c) P(True)-style confidence for emotion recognition;
- (d) self-consistency combined with verification for emotion.

## Searches (arXiv API, 2026-11-24)

21 field × method queries:
- emotion × {verification + LLM, self-verification, verify + candidate + multimodal, rerank, re-rank, reranking + language model, "yes or no", P(True), generate-then-verify, self-consistency + multimodal, minimum Bayes risk, candidate labels + verif, multi-label + zero-shot + verif, verifier + test-time, verifier + multimodal, binary question, training-free + MLLM, test-time scaling, best-of-N};
- affective × verifier;
- "open-vocabulary" × emotion.

About 130 distinct results screened by title; 13 abstracts read; 6 full texts read.

**Not yet done:** OpenAlex / Semantic Scholar sweep, snowballing, ACM / IEEE proceedings. To finish before P18 is registered.

## Closest papers (full text read) and verdicts

| Paper | What it does | Verdict |
|---|---|---|
| **Decoding Ambiguous Emotions with Test-Time Scaling in Audio-Language Models** (arXiv 2602.03873, 02/2026) | Ambiguous SER (IEMOCAP, MSP-Podcast, CREMA-D) as emotion distributions. Five test-time strategies: CoT; beam / best-of-N selected by log-likelihood; log-likelihood-weighted Dirichlet-mixture aggregation of beams; an external verifier (GPT-4o) that scores whole responses; a verifier-weighted combination (§4.2) | **Partial overlap**, for both CONSTRUCT and P18: test-time sampling, aggregation and a verifier in affective computing. Differences: closed-set label distributions (not open-vocabulary sets); the verifier is an external stronger model scoring whole responses, not the same model judging each candidate label; no metric-aware set construction. **Must be cited and positioned in the paper.** |
| **HyDRA: Hybrid-evidential Deductive Reasoning in OV-MER** (arXiv 2603.16463) | Propose–Verify–Decide over situational hypotheses inside the model's reasoning, trained with GRPO (HumanOmni-0.5B) | Partial: "verify" happens inside the reasoning and is learned by RL; it does not score candidate labels at test time. |
| **MVEI & EmObserver: Emotion Statement Judgement** (arXiv 2607.21061) | Image affect recast as yes/no judgement of emotion statements; benchmark plus a fine-tuned MLLM | Partial: yes/no judgement of emotion statements, but for images, trained, and used as the task itself, not as a decoder for generative MER. |
| **An Agentic Approach with Evidence-Grounded Verification for Multimodal Emotion Reasoning in Conversation** (arXiv 2610.31685) | Frozen MLLM agents, an inline critic that can request one revision, offline evidence / entailment checks to attribute failures and patch prompts; MER2025 Track 3, EW metric | Partial: verification is used for failure attribution and prompt repair, not as per-label probabilities in decoding. |
| **Emotional Rationale Verifier** (arXiv 2510.23506, AAAI 2026) | A verifier of rationale–emotion consistency, used as an RL reward | Not the same: a training-time reward. |
| **BiCFlow-MER** (arXiv 2609.27615) | Audio–text conditional flow; candidates "verified" geometrically in an emotion space | Not the same. |
| MER2024 OV report (arXiv 2408.11286) | "Multi-model co-judgment": splicing the outputs of several models | Not the same. |

Abstract only: DAN (2609.36782, training-free contrastive visual pruning) and HMER (2609.11154, hallucination-aware logit rectification), both training-free but other mechanisms; EmotionHallucer (2505.11405) uses binary questions for evaluation only.

**Other-domain precedents** (to cite; C4 allows them): P(True) self-evaluation of language models; self-verification of LLM answers; verifier-guided best-of-N.

## Status

No scoop found in this arXiv sweep.

2602.03873 is now a required related-work entry for the whole F1 line, CONSTRUCT included: it is test-time scaling for emotion with aggregation and a verifier, although closed-set and speech-only.
