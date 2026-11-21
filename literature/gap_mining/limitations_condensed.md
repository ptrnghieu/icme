
### 2610.31685 | 2026-10-03 | An Agentic Approach with Evidence-Grounded Verification for Multimodal Emotion Reasoning in Conversation
- Two limitations are worth noting.
- First, the verification layer audits only what a draft actually cites (Section 3.2.1): a wrong prediction whose citations pass every deterministic and entailment check is labeled irreducible (Section 3.2.2) and cannot be attributed to any agent or repaired, so the repair loop’s reach is capped by how much of the error rate is attributable in the first place, not by how well it acts on the errors it does catch.
- Future work could turn the same per-agent attributions into supervision for finetuning the responsible agent, and the per-claim deterministic and entailment scores into a processlevel reinforcement-learning reward, extending the current training-free loop to drive weight updates directly.
- Question: Does the paper discuss the limitations of the work performed by the authors?
- Answer: [Yes] Justification: A dedicated Limitations section (§5) discusses the irreducible class of errors that verification cannot attribute or repair, and the unquantified inference-time cost of the multi-call pipeline relative to the single-call baselines.
- Guidelines: • The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

### 2609.30238 | 2026-09-24 | SemMSA: Latent Semantic-Aided Robust Multimodal Sentiment Analysis with Incomplete Data
- While SemMSA demonstrates strong performance across a variety of MSA benchmarks, several limitations remain.
- Question: Does the paper discuss the limitations of the work performed by the authors?
- Answer: [Yes] Justification: The limitations of the work are discussed in the Limitations section of the Appendix.
- Guidelines: • The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- • The authors are encouraged to create a separate "Limitations" section in their paper.
- • The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs.

### 2609.16581 | 2026-09-15 | EmoPhone: A Multi-Wave Dataset for In-the-Wild Mobile and Wearable Affect Sensing
- These results indicate that user-level adaptation does not automatically translate into wave-level robustness, and that no single score adequately summarizes performance across within-user, cross-user, and cross-wave settings.
- 6.3 Limitations The dataset should also be read with clear but proportionate boundaries.
- Finally, the eight custom (non-PANAS) affect-intensity items appear only in D-3, so fine-grained emotion analyses cannot yet be carried out symmetrically across all three waves.
- Taken together, these findings suggest that performance remains modest and the observed cross-wave differences cannot be attributed to a single cause.
- Question: Does the paper discuss the limitations of the work performed by the authors?
- Answer: [Yes] Justification: The paper includes an explicit limitations section discussing the studentcentered cohort, Android-only sensing, imperfect cross-wave comparability, partial modality overlap, and the fact that richer affect labels are available only in D-3.

### 2609.11247 | 2026-09-10 | The Illusion of Balanced Multimodal Sentiment Analysis: Beyond the Limits of Optimization-Based Methods
- Their core limitation is clear: they treat losses and gradients as indicators of discriminative value.
- Our work provides both a diagnostic foundation, establishing what current optimization methods cannot achieve and why, and a research agenda pointing toward discriminative metalearning frameworks that separate encoder optimization from fusion weight optimization.

### 2609.10950 | 2026-09-10 | Robust Multimodal Sentiment Analysis with Incomplete Modalities via Semantic-aware Completeness based Reconstruction
- Limitations Although TCMR consistently outperforms existing reconstruction-based MSA models, several limitations remain to be addressed in future work.
- Future work could mitigate this issue by adopting confidenceaware filtering, using only high-confidence TPSC predictions as pseudo-labels while relying on complementary strategies such as LDSC for less confident samples.
- Although TCMR consistently outperforms existing reconstruction-based MSA models, several limitations remain to be addressed in future work.

### 2609.11164 | 2026-09-10 | Multimodal Temporal Modeling for Continuous Group Emotion Recognition in Multi-party Dialogues
- This comparison does not establish a general limitation of LLMs: the current multimodal interface and amount of training data may not fully exploit their capacity, while language-based reasoning and explanation remain promising uses of LLMs for this task.
- Several limitations qualify these conclusions.
- These findings, together with the limitations discussed above, suggest three directions for future work.
- Second, identifying which visual cues provide reliable complementary information to audio remains an open problem.

### 2609.09672 | 2026-09-09 | SEA-SpeechBench: A Large-Scale Multitask Benchmark for Speech Understanding Across Southeast Asia
- By surfacing these limitations, SEA-S PEECH B ENCH seeks to establish a rigorous baseline for developing temporally robust, linguistically inclusive, and practically deployable speech technologies for Southeast Asia’s diverse communities.

### 2609.05281 | 2026-09-04 | KanAdapter: A Kolmogorov-Arnold Network-based Plug-and-Play Module for Efficient Fine-tuning of Foundation Speech Models
- We leave these explorations for future work.
- Limitations Our study has several limitations.
- Our study has several limitations.

### 2608.30563 | 2026-08-31 | Modality Disentangled Learning for Incomplete Multimodal Emotion Recognition: A Primitive Memory Distillation Perspective
- Although PriMD achieves good robustness under various incomplete multimodal settings, several limitations remain.
- To reduce this dependence, future work may combine cross-modal self-supervised pretraining or semisupervised learning to make better use of incomplete samples.
- PriMD does not yet explicitly model the dependence between missing states at different time steps.
- Future work will explore time-aware retrieval, modality-quality gating, historical primitive caching, and online memory updates.
- Multimodal sentiment analysis: A systematic review of history, datasets, multimodal fusion methods, applications, challenges and future directions.

### 2608.28966 | 2026-08-29 | When Vocal Tone and Literal Meaning Diverge: An Acoustic-Semantic Incongruity Study for Large Audio-Language Models
- We further show that TTS-synthesized data may offer a promising direction to mitigate this limitation.
- Limitations Methodology limitations: The primary goal of this study is to provide a baseline evaluation of LALMs on the acoustic–semantic inconsistent task, and certain limitations follow from this scope.
- Dataset quality and evaluation limitations: We acknowledge several limitations of the proposed dataset.
- Methodology limitations: The primary goal of this study is to provide a baseline evaluation of LALMs on the acoustic–semantic inconsistent task, and certain limitations follow from this scope.
- Generalizability and scope limitations: Our dataset is generated from acted emotion references, which are often exaggerated and easier to identify than spontaneous speech.
- To assess generalizability, we evaluate SFT models on MELD (TV dialogue) and LISTEN (acted and spontaneous subsets), finding some transferability, but potential limitations remain.

### 2608.29035 | 2026-08-29 | EmoLASP: Emotion Recognition with Language Models and Answer Set Programming
- However, qualitative analysis reveals a limitation of LLM-based approaches: they struggle to accurately interpret extreme emotions in brief utterances or those relying on metaphorical language.
- Future work could explore the use of these models as baselines by mapping their categorical predictions to VAD scores.
- Our study has several limitations, which concern the data, the framework, and our evaluation.
- Second, the framework itself has limitations.
- In future work, we plan to explore more elaborate ASP implementations.
- Our ablation studies in Appendix D.2 show that the results barely change across a range of these values, and that weighting the history by speaker identity does not help either.

### 2608.08702 | 2026-08-09 | SRE-FER: Regional residual evidence learning for mitigating local evidence dilution in fine-grained facial expression recognition
- The fixed regional partition, unmatched alternative readouts, and unprofiled Full path remain limitations.

### 2607.18336 | 2026-07-19 | EmoEUS: Uncertainty Supervision for Multimodal Emotion Recognition in Conversation
- In future work, we will enhance the uncertainty estimation module to better capture cross-speaker emotional dynamics in real-time settings.

### 2607.12787 | 2026-07-14 | Do We Really Need Multimodal Emotion Language Models Larger Than 1B Parameters?
- These results suggest that efficient generative MER does not require large deployment models, and that preserving representation geometry is a promising direction for compact multimodal reasoning systems.

### 2606.27536 | 2026-06-25 | Learning from Annotation Uncertainty: Entropy-Aware Curriculum for Speech Emotion Recognition
- However, gains in the high-entropy region remain limited.
- Thus, uncertainty-aware supervision may help preserve performance under annotation ambiguity, although any gains remain constrained by the limitations of hard categorical evaluation in high-entropy regions.
- Entropy-stratified results show that high-entropy utterances remain challenging under hard categorical evaluation.

### 2606.06200 | 2026-06-04 | Learning Emotion-discriminative Representations for Zero-Shot Cross-lingual Speech Emotion Recognition
- Future work will explore (1) stronger crosslingual alignment methods, and (2) incorporating multi-modal information to further approach the upper-bound performance.

### 2605.25409 | 2026-05-25 | MTLLFM: Multimodal-Temporal Laughter Localization: UR-FUNNY-Temporal and SMILE-Temporal Benchmarks with an Adaptive Multimodal Fusion Model
- Crucially, while our model never uses these annotations during training, we release them in full to enable a broad range of future research directions: • Supervised and semi-supervised localization: our annotations enable training fully-supervised TAL models as strong baselines, and studying how few temporal labels can bootstrap weakly-supervised methods.

### 2605.00871 | 2026-04-24 | NAKUL-Med: Spectral-Graph State Space Models with Dynamics Kernels for Medical Signals
- Future work includes bilateral motor disambiguation via asymmetry-aware graph structure and emotion classification improvements through multimodal fusion with facial expressions.

### 2604.18460 | 2026-04-20 | Learning Invariant Modality Representation for Robust Multimodal Learning from a Causal Inference Perspective
- While CmIR demonstrates strong performance and robustness, our work has certain limitations.
- Future work could explore more sophisticated environment generation strategies or incorporate realworld multi-environment datasets.
- These limitations, however, do not undermine the core theoretical contributions or the empirical effectiveness of the proposed framework.

### 2604.17647 | 2026-04-19 | Prosody as Supervision: Bridging the Non-Verbal--Verbal for Multilingual Speech Emotion Recognition
- Limitations and Future Work The current evaluation across multiple publicly available benchmarks demonstrates the effectiveness of proposed framework for non-verbal-toverbal transfer in multilingual SER.

### 2604.12579 | 2026-04-14 | EEG-Based Multimodal Learning via Hyperbolic Mixture-of-Curvature Experts
- While our framework shows promising results, several limitations and future directions warrant discussion.
- The current approach focuses mainly on hyperbolic geometry and modalities with hierarchical structures; future work could explore other modalities with different structural priors.
- The computational stability and overhead of hyperbolic operations may suggest potential directions for future research.
- Although the unified manifold strategy for fusion is empirically effective, richer fusion and transport mechanisms may be explored in future work.

### 2604.09162 | 2026-04-10 | Persona-E$^2$: A Human-Grounded Dataset for Personality-Shaped Emotional Responses to Textual Events
- Limitations We introduce Persona-E2 , a large-scale dataset explicitly grounded in personality traits to model reader-based emotional variation.
- However, the dataset has several limitations.

### 2604.01916 | 2026-04-02 | SURE: Synergistic Uncertainty-aware Reasoning for Multimodal Emotion Recognition in Conversations
- In future work, we plan to explore more adaptive expert-sharing strategies and extend the framework to broader multimodal dialogue understanding tasks.

### 2603.21050 | 2026-03-22 | ERM-MinMaxGAP: Benchmarking and Mitigating Gender Bias in Multilingual Multimodal Speech-LLM Emotion Recognition
- Through evaluation on MELD- ST across English, Japanese, and German, we showed that gender disparity is highly dependent on both language and input modality, and that multimodal input does not reliably reduce bias even when it improves SER performance.

### 2603.19718 | 2026-03-20 | BALM: A Model-Agnostic Framework for Balanced Multimodal Learning under Imbalanced Missing Rates
- Future work will explore broader multimodal tasks and large-scale pretrained backbones.

### 2603.18462 | 2026-03-19 | AlignMamba-2: Enhancing Multimodal Fusion and Sentiment Analysis with Modality-Aware Mamba
- Our work not only presents a powerful solution for multimodal learning and sentiment analysis but also highlights a critical direction for future research: exploring linear-complexity alignment metrics or sparse mechanisms to achieve unified efficiency across both training and inference phases, ensuring discriminative multimodal representations with minimal computational cost.

### 2603.15020 | 2026-03-16 | MER-Bench: Benchmarking Multimodal Emotional Reappraisal for Image-Text Memes
- These findings position MER as a challenging multimodal reasoning problem beyond conventional image editing or sentiment transfer, and we hope MER-Bench will support future research on controllable affect-aware generation.
- Limitations MER-Bench focuses on memes, leaving dynamic formats such as GIFs, short videos, and audiovisual memes unexplored.
- It also does not explicitly address multilingual or cross-cultural variation, despite the influence of language, culture, and community conventions on meme interpretation and emotional reappraisal.
- Future work will extend MER toward multilingual, multicultural, and temporally grounded settings.
- Although downstream risks cannot be eliminated, MER-Bench is designed to promote safer affective transformation.

### 2603.12693 | 2026-03-13 | HSEmotion Team at ABAW-10 Competition: Facial Expression Recognition, Valence-Arousal Estimation, Action Unit Detection and Fine-Grained Violence Classification
- While our experiments demonstrate the benefits of this pragmatic approach, future directions include stronger self-supervised pretraining, domain-adaptive calibration, and richer temporal models to further improve long-range consistency and uncertainty quantification for safety-sensitive applications.

### 2603.11095 | 2026-03-11 | Multimodal Self-Attention Network with Temporal Alignment for Audio-Visual Emotion Recognition
- While this work focuses on controlled benchmarks, extending the proposed framework to large-scale in-the-wild datasets remains an important direction for future work.

### 2602.23802 | 2026-02-27 | EMO-R3: Reflective Reinforcement Learning for Emotional Reasoning in Multimodal Large Language Models
- Building upon this foundation, future research may further explore the generalization of emotion recognition with reasoning in more complex multimodal scenarios, including sequential or interactive task settings.

### 2602.19585 | 2026-02-23 | Tri-Subspaces Disentanglement for Multimodal Sentiment Analysis
- In the future, we will expand TSD to a wider range of human-computer interaction scenarios.

### 2602.18802 | 2026-02-21 | Multi-Channel Speech Enhancement for Cocktail Party Speech Emotion Recognition
- Future research will focus on developing multi-task systems that combine speech and emotion recognition in complex acoustic environments.

### 2602.01547 | 2026-02-02 | Attention-weighted Centered Kernel Alignment for Knowledge Distillation in Large Audio-Language Models Applied to Speech Emotion Recognition
- We will explore applying PL-Distill to more tasks in the future.

### 2602.00811 | 2026-01-31 | MissMAC-Bench: Building Solid Benchmark for Missing Modality Issue in Robust Multimodal Affective Computing
- Limitations Despite its comprehensive design, MissMACBench has several limitations as the future improvement direction.
- Third, although the benchmark has evaluated several representative encoder-based models in MSA, future work should include more advanced architectures, particularly decoder-based multimodal large language models, to better understand their robustness under missing modality conditions.
- Despite its comprehensive design, MissMACBench has several limitations as the future improvement direction.

### 2601.18010 | 2026-01-25 | AmbER$^2$: Dual Ambiguity-Aware Emotion Recognition Applied to Speech and Text
- Finally, the framework is generalizable across modalities and languages, and future work will apply it to multimodal emotion recognition and multilingual, cross-cultural datasets, enabling a deeper examination of ambiguity and variability across diverse annotation contexts.

### 2601.16240 | 2026-01-21 | Test-Time Adaptation for Speech Emotion Recognition
- This study provides a foundational investigation and offers critical insights for future research into more robust and adaptive SER systems.

### 2601.14620 | 2026-01-21 | Scaling Ambiguity: Augmenting Human Annotation in Speech Emotion Recognition with Audio-Language Models
- Limitations include reliance on specific models and reduced effectiveness on acted emotions in IEMOCAP.

### 2601.15395 | 2026-01-21 | Beyond Fixed Psychological Personas: State Beats Trait, but Language Models are State-Blind
- We release the dataset and code to support this goal.2 5 Limitations.
- We acknowledge several limitations.
- Human annotation of these profiles was not conducted, as reliable psychological construct rating requires specialist training to achieve consistent inter-rater agreement; future work should establish criterion validity through ecological momentary assessment with consenting users.
- Our context operationalization (subreddits) conflates topic, audience, and community norms; future work could disentangle these factors.
- Future work could extend this approach by sampling users across a larger number of contexts (k=10+), enabling more precise variance decomposition and finer-grained analysis of which context types drive the mos

### 2601.13659 | 2026-01-20 | Temporal-Spatial Decouple before Act: Disentangled Representation Learning for Multimodal Sentiment Analysis
- In future work, we plan to extend TSDA to the field of human-computer interaction.

### 2601.12591 | 2026-01-18 | SmoothCLAP: Soft-Target Enhanced Contrastive Language\--Audio Pretraining for Affective Computing
- Nevertheless, the influence of the training data remains an open question, with both quantity and diversity expected to be another key factor.

### 2512.16485 | 2025-12-18 | Smile on the Face, Sadness in the Eyes: Bridging the Emotion Gap with a Multimodal Dataset of Eye and Facial Behaviors
- In the future, we plan to extend EMER with more diverse and ecologically valid scenarios and release it to encourage further research in emotion-related tasks.

### 2512.15250 | 2025-12-17 | Leveraging Foundational Models and Simple Fusion for Multi-modal Physiological Signal Analysis
- Despite the promising results, our work has several limitations.
- Future work will explore more advanced fusion mechanisms, extend evaluation to additional downstream tasks, and investigate new strategies for mitigating data scarcity, such as weak supervision or synthetic data augmentation.

### 2512.04551 | 2025-12-04 | Multi-Loss Learning for Speech Emotion Recognition with Energy-Adaptive Mixup and Frame-Level Attention
- Future work will extend it to crosslingual settings, multimodal cues, and more advanced adaptive augmentations.

### 2512.03521 | 2025-12-03 | Cross-Space Synergy: A Unified Framework for Multimodal Emotion Recognition in Conversation
- Future work will explore broader multimodal settings and pursue a more expressive fusion structure with stronger conflict-resilient optimization.

### 2512.01442 | 2025-12-01 | PSA-MF: Personality-Sentiment Aligned Multi-Level Fusion for Multimodal Sentiment Analysis
- Future work could extend personalized sentiment analysis to other modalities and explore new fusion strategies.

### 2511.23178 | 2025-11-28 | HPSU: A Benchmark for Human-Level Perception in Real-World Spoken Speech Understanding
- Our analysis identified current limitations in deep inferential reasoning, robustness against biases, and nuanced cross-lingual comprehension.

### 2511.22188 | 2025-11-27 | ARPGNet: Appearance- and Relation-aware Parallel Graph Attention Fusion Network for Facial Expression Recognition
- 5.2 Limitations and Future Work Despite the promising results achieved by the proposed ARPGNet framework in integrating appearance and relation representations for facial expression recognition, several limitations warrant discussion and suggest avenues for future research.
- To mitigate this, future work could explore the integration of more advanced representation learning techniques, such as Masked Auto-Encoders (MAEs) [85] or Multimodal Large Models (MLMs) [37].
- To address this limitation, investigating unsupervised or self-supervised domain adaptation techniques [86], [87], particularly Test-Time Adaptation (TTA) strategies, could be a fruitful direction.
- Despite the promising results achieved by the proposed ARPGNet framework in integrating appearance and relation representations for facial expression recognition, several limitations warrant discussion and suggest avenues for future research.

### 2511.14249 | 2025-11-18 | Towards Authentic Movie Dubbing with Retrieve-Augmented Director-Actor Interaction Learning
- In future work, we plan to explore more expressive modeling of director–actor interactions in real dubbing workflows, including controllable aspects such as timbre, speaking rate, and prosody.

### 2511.11002 | 2025-11-14 | EmoVid: A Multimodal Emotion Video Dataset for Emotion-Centric Video Understanding and Generation
- We will continue to explore these directions in our future work.

### 2511.10325 | 2025-11-13 | TMDC: A Two-Stage Modality Denoising and Complementation Framework for Multimodal Sentiment Analysis with Missing and Noisy Modalities
- Future work will aim to reduce this redundancy to further improve efficiency.

### 2511.07061 | 2025-11-10 | Do LLMs Feel? Teaching Emotion Recognition with Prompts, Retrieval, and Curriculum Learning
- In future work, we plan to further explore stronger LLMs, more efficient prompting paradigms, and advanced curriculum designs to enhance emotional reasoning and improve the robustness and generalization of ERC systems.

### 2511.02794 | 2025-11-04 | When One Modality Sabotages the Others: A Diagnostic Lens on Multimodal Reasoning
- Beyond emotion recognition, the proposed approach offers a general scaffold for auditing multimodal reasoning systems and guiding future work on calibration, conflict resolution, and interpretable fusion.

### 2510.05749 | 2025-10-07 | MSF-SER: Enriching Acoustic Modeling with Multi-Granularity Semantics for Speech Emotion Recognition
- In future work, we plan to extend this design to cross-lingual scenarios and explore its integration with additional modalities such as visual signals.

### 2510.03060 | 2025-10-03 | Semantic Differentiation in Speech Emotion Recognition: Insights from Descriptive and Expressive Speech Roles
- This differentiation can inform future research in emotion detection.
- In other applications, it might be more suitable to leave it to the users to express their emo- This study, while providing valuable insights into the segmentation of speech for emotion recognition, has limitations.
- However, future work could explore whether jointly modeling text and acoustic features, perhaps guided by semantic segmentation, might uncover latent prosodic patterns aligned with specific se- mantic roles.

### 2510.00725 | 2025-10-01 | DEAP DIVE: Dataset Investigation with Vision transformers for EEG evaluation
- Pairing subsets that achieved a high accuracy with additional channels does not always lead to better performances.
- The underlying mechanisms of this non-superpositional behavior and the factors contributing to the amplification of accuracy across certain channels remain open questions for further investigation in the field of explainable AI.

### 2509.15540 | 2025-09-19 | Beyond Words: Enhancing Desire, Emotion, and Sentiment Recognition with Non-Verbal Cues
- Although we thoroughly explore non-verbal cues in images through a masked image modeling and our proposed method demonstrated effectiveness on the MSED dataset, several limitations remain: (1) Capturing fine-grained local features is still insufficient.
- Effectively leveraging such terms, integrating and aligning them with visual information represents a promising direction for future research.

### 2508.19533 | 2025-08-27 | Emotion Transfer with Enhanced Prototype for Unseen Emotion Recognition in Conversation
- In future work, our goal is to further optimize prototype representations.
- Additionally, our approach focuses solely on the text modality and does not incorporate multi-modal information, such as facial expressions, which could provide valuable additional information.

### 2508.18655 | 2025-08-26 | Empathy Omni: Enabling Empathetic Speech Response Generation through Large Language Models
- Future work will focus on improving sensitivity to nuanced emotions.

### 2508.18092 | 2025-08-25 | Speech-Based Depressive Mood Detection in the Presence of Multiple Sclerosis: A Cross-Corpus and Cross-Lingual Study
- Future work should explore longer segments to optimise textbased models, building on this study’s foundation.
- A limitation of this contribution arises from the use of different languages, recording conditions, and depression assessments.
- In this paper, we cannot definitively differentiate the extent to which the drop in model performance when evaluated on the MS population is influenced by language differences, recording conditions or the presence of MS itself.
- Future work should also explore different stages of MS –this study focused on low-disability patients– and account for other co-morbidities in MS, like fatigue and cognitive decline, which may also influence speech.
- This study, despite its limitations, represents a step forward towards the integration and generalisability of speech-based depression detection methods.

### 2508.10644 | 2025-08-14 | Conditional Information Bottleneck for Multimodal Fusion: Overcoming Shortcut Learning in Sarcasm Detection
- In the future, we aim to refine MCIB into an easily integrable plug-in for various backbone models in multimodal sentiment analysis, thereby boosting their performance.

### 2508.07282 | 2025-08-10 | Lessons Learnt: Revisit Key Training Strategies for Effective Speech Emotion Recognition in the Wild
- However, our model exhibits limited improvements in categorical emotion, arousal, and dominance prediction, suggesting that a more generalized SER framework remains an open challenge.
- Future work should focus on a deeper analysis and more comprehensive modeling approaches to improve SER performance in naturalistic, real-world conditions.

### 2508.06564 | 2025-08-06 | Grounding Emotion Recognition with Visual Prototypes: VEGA -- Revisiting CLIP in MERC
- Beyond accuracy, our approach offers a modular and psychologically inspired paradigm for multimodal emotion recognition, paving the way for future research in semantically structured and human-aligned affective computing.

### 2508.04723 | 2025-08-05 | Wearable Music2Emotion : Assessing Emotions Induced by AI-Generated Music through Portable EEG-fNIRS Fusion
- We address three key limitations of existing studies: (1) restricted stimulus diversity, (2) single-modality of signals, and (3) lack of portability.

### 2508.01644 | 2025-08-03 | DRKF: Decoupled Representations with Knowledge Fusion for Multimodal Emotion Recognition
- Despite the promising results achieved by the proposed DRKF model on bimodal emotion recognition tasks, certain limitations remain.
- In future work, we plan to further explore the adaptability and scalability of DRKF in more complex multimodal input settings, such as those involving video, speech, and text, to better address the demands of real-world multimodal emotion recognition applications.

### 2507.18929 | 2025-07-25 | MGHFT: Multi-Granularity Hierarchical Fusion Transformer for Cross-Modal Sticker Emotion Recognition
- Inspired by the way humans understand the emotion of stickers, this paper proposes a novel Multi-granularity Hierarchical Fusion Transformer to overcome the limitation of existing pre-trained visual models for sticker emotion recognition.
- We hope this work can serve as a valuable reference and inspiration for future research in sticker emotion analysis.

### 2507.15765 | 2025-07-21 | Learning from Heterogeneity: Generalizing Dynamic Facial Expression Recognition via Distributionally Robust Optimization
- In future work, we plan to extend this framework to multi-modal expression analysis and investigate deployment-friendly variants for real-world applications.

### 2507.13626 | 2025-07-18 | Unifying Listener Scoring Scales: Comparison Learning Framework for Speech Quality Assessment and Continuous Speech Emotion Recognition
- For future work, we suggest investigating the feasibility of establishing a unified listener scoring scale across different dataset domains.

### 2506.21017 | 2025-06-26 | Multimodal Prompt Alignment for Facial Expression Recognition
- Future work will explore further integration of multimodal cues and the extension of our framework to other fine-grained visual recognition tasks.

### 2506.19887 | 2025-06-24 | MATER: Multi-level Acoustic and Textual Emotion Representation for Interpretable Speech Emotion Recognition
- Future work will focus on emotion-specific feature selection and extending MATER to diverse SER datasets, advancing toward a more robust real-world SER system.

### 2506.18716 | 2025-06-23 | Multi-modal Anchor Gated Transformer with Knowledge Distillation for Emotion Recognition in Conversation
- Future work will explore extending MAGTKD to handle more complex multi-modal scenarios, such as incorporating dynamic contextual information in real-time conversations or addressing challenges posed by highly imbalanced datasets.

### 2506.09556 | 2025-06-11 | MEDUSA: A Multimodal Deep Fusion Multi-Stage Training Framework for Speech Emotion Recognition in Naturalistic Conditions
- Future research directions include integrating forced alignment and speaker information to enrich the feature space, and exploring modality-agnostic augmentations for improved generalization.

### 2506.08717 | 2025-06-10 | Multi-Teacher Language-Aware Knowledge Distillation for Multilingual Speech Emotion Recognition
- In future studies, we will explore heterogeneous teachers, alternative similarity metrics, and broader linguistic and cultural contexts to develop a more robust and versatile SER system.

### 2506.03403 | 2025-06-03 | HYFuse: Aligning Heterogeneous Speech Pre-Trained Representations in Hyperbolic Space for Speech Emotion Recognition
- These findings highlight the potential of heterogeneous representation fusion in advancing SER and also as a reference for future research exploring such heterogeneous fusion.

### 2506.02230 | 2025-06-02 | Towards Machine Unlearning for Paralinguistic Speech Processing
- To aid future research, we provide “cookbook recipes” and as a part of this we recommend the use of transformer-based downstream with TRILLsson features for MU in PSP as they offer the best performance retention and validated through our experiments.

### 2506.01138 | 2025-06-01 | PARROT: Synergizing Mamba and Attention-based SSL Pre-Trained Models via Parallel Branch Hadamard Optimal Transport for Speech Emotion Recognition
- Our study will act as a reference for future research towards heterogeneous fusion of PTMs for SER.

### 2505.23962 | 2025-05-29 | Can Emotion Fool Anti-spoofing?
- For future work, we aim to enhance anti-spoofing resilience to emotion-targeted attacks across diverse emotional states and TTS models while expanding EmoSpoof-TTS.

### 2505.23236 | 2025-05-29 | Towards LLM-Empowered Fine-Grained Speech Descriptors for Explainable Emotion Recognition
- Future research will focus on exploring the differences in fine-grained emotion descriptors when different people express the same emotion.

### 2505.20511 | 2025-05-26 | Multimodal Emotion Recognition in Conversations: A Survey of Methods, Trends, Challenges and Prospects
- However, several limitations should be acknowledged to contextualize the scope of our work.
- We view these open questions as productive entry points for future research and hope this survey supports ongoing efforts to develop more advanced MERC systems.

### 2505.20007 | 2025-05-26 | Improving Speech Emotion Recognition Through Cross Modal Attention Alignment and Balanced Stacking Model
- Future work will involve analyzing the use of data augmentation and exploring the emotion adaptation of SSL models before applying the proposed cross-modal architecture.

### 2505.19693 | 2025-05-26 | EmoSphere-SER: Enhancing Speech Emotion Recognition Through Spherical Representation with Auxiliary Classification
- Future research can further explore refining region partitioning strategies and extending this framework to other affective computing tasks.

### 2505.16220 | 2025-05-22 | Meta-PerSER: Few-Shot Listener Personalized Speech Emotion Recognition via Meta-learning
- This limitation may hinder the system’s ability to accurately capture the natural context of individual emotion perception.
- For future research, we plan to explore the applicability of Meta-PerSER to other subjective learning tasks beyond SER, such as hate speech detection and customer experience recognition.

### 2505.14449 | 2025-05-20 | Mitigating Subgroup Disparities in Multi-Label Speech Emotion Recognition: A Pseudo-Labeling and Unsupervised Learning Approach
- While our approach demonstrates robust improvements in mitigating subgroup disparities, several limitations warrant discussion.
- For future work, we will explore more possible bias attributes like content [34], language [35], or speaking style [36].

### 2505.11051 | 2025-05-16 | CAMEO: Collection of Multilingual Emotional Speech Corpora
- Future work will focus on several key areas.
- Second, we intend to evaluate new models to make CAMEO a continuously developing resource for advancing SER research.

### 2504.19549 | 2025-04-28 | DEEMO: De-identity Multimodal Emotion Recognition and Reasoning
- We hope that our dataset and framework can inspire future research in privacy-preserving emotion understanding and ethical human-centered AI.

### 2504.14882 | 2025-04-21 | Some Optimizers are More Equal: Understanding the Role of Optimizers in Group Fairness
- Question: Does the paper discuss the limitations of the work performed by the authors?
- Guidelines: • The answer NA means that the paper has no limitation while the answer No means that the paper has limitations, but those are not discussed in the paper.
- • The authors are encouraged to create a separate "Limitations" section in their paper.
- • The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs.
- • If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- • While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper.

### 2502.10973 | 2025-02-16 | Akan Cinematic Emotions (ACE): A Multimodal Multi-party Dataset for Emotion Recognition in Movie Dialogues
- While the Akan Cinematic Emotions (AkaCE) dataset represents a significant advancement in multimodal emotion recognition research, particularly for African languages, there are several limitations to acknowledge.
- One limitation of this work is that the dataset focuses exclusively on the Akan language.
- Another limitation lies in the domain of the dataset, which is derived from movie dialogues.

### 2502.09993 | 2025-02-14 | Navigating Label Ambiguity for Facial Expression Recognition in the Wild
- Furthermore, while NLA is designed for FER, its ability to address label ambiguity in noisy or imbalanced data indicates promising applicability to a broader range of tasks, which we leave as an avenue for future work.
