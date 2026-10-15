
# The standard method of post training is called Supervised Fine-Tuning (SFT). Other methods include preference optimization (DPO, ORPO), distillation and Reinforcement Learning (RL) (GRPO, GSPO), where an "agent" learns to make decisions by interacting with an environment and receiving feedback in the form of rewards or penalties.

# Example fine-tuning or RL use-cases:Enables LLMs to predict if a headline impacts a company positively or negatively.Can use historical customer interactions for more accurate and custom responses.Fine-tune LLM on legal texts for contract analysis, case law research, and compliance.

# Fine-tuning can replicate all of RAG's capabilities

# LoRA and QLoRA are efficient shortcuts used to train Large Language Models (LLMs) like Llama 70B by updating only 1% of the weights and shrinking data to save 75% memory. Think of it as updating a massive encyclopedia by adding small sticky notes instead of rewriting the entire book.

# The Encyclopedia Analogy
# • Model Weights: These are the 70 billion numbers acting as the facts inside the encyclopedia.
# • LoRA (Sticky Notes): Instead of editing all 70 billion numbers, you attach tiny sticky notes containing just the new updates.
# • QLoRA (Font Shrinking): You print the encyclopedia in a much smaller font to save shelf space, then add the sticky notes.

# Fine-tuning an LLM customizes its behavior, deepens its domain expertise, and optimizes its performance for specific tasks. By refining a pre-trained model (e.g. Llama-3.1-8B) with specialized data, you can:

# Update Knowledge – Introduce new, domain-specific information that the base model didn’t originally include.

# Customize Behavior – Adjust the model’s tone, personality, or response style to fit specific needs or a brand voice.

# Optimize for Tasks – Improve accuracy and relevance on particular tasks or queries your use-case requires.

# Think of fine-tuning as creating a specialized expert out of a generalist model. Some debate whether to use Retrieval-Augmented Generation (RAG) instead of fine-tuning, but fine-tuning can incorporate knowledge and behaviors directly into the model in ways RAG cannot. In practice, combining both approaches yields the best results - leading to greater accuracy, better usability, and fewer hallucinations.

# Some of the real world application of fine-tunning
# Sentiment Analysis for Finance
# Customer Support Chatbots
# Legal Document Assistance

# Fine-Tuning vs. RAG: What’s the Difference? Fine-tuning can do mostly everything RAG can - but not the other way around. During training, fine-tuning embeds external knowledge directly into the model. This allows the model to handle niche queries, summarize documents, and maintain context without relying on an outside retrieval system. That’s not to say RAG lacks advantages as it is excels at accessing up-to-date information from external databases. It is in fact possible to retrieve fresh data with fine-tuning as well, however it is better to combine RAG with fine-tuning for efficiency.

# Fine-Tuning deeply integrates domain knowledge into the model.This makes it highly effective at handling structured, repeat tasks

# Does Fine-Tuning Add New Knowledge to a Model?
# Yes - it absolutely can. A common myth suggests that fine-tuning doesn’t introduce new knowledge, but in reality it does. If your fine-tuning dataset contains new domain-specific information, the model will learn that content during training and incorporate it into its responses. In effect, fine-tuning can and does teach the model new facts and patterns from scratch

# Is RAG Always Better Than Fine-Tuning?Not necessarily. Many assume RAG will consistently outperform a fine-tuned model, but that’s not the case when fine-tuning is done properly. In fact, a well-tuned model often matches or even surpasses RAG-based systems on specialized tasks. Claims that “RAG is always better” usually stem from fine-tuning attempts that weren’t optimally configured - for example, using incorrect LoRA parameters or insufficient training.

# Is Fine-Tuning Expensive?Not at all! While full fine-tuning or pretraining can be costly, these are not necessary (pretraining is especially not necessary). In most cases, LoRA or QLoRA fine-tuning can be done for minimal cost. In fact, with Unsloth’s free notebooks for Colab or Kaggle, you can fine-tune models without spending a dime. Better yet, you can even fine-tune locally on your own device.


# Why You Should Combine RAG & Fine-Tuning Instead of choosing between RAG and fine-tuning, consider using both together for the best results. Combining a retrieval system with a fine-tuned model brings out the strengths of each approach. Here’s why:

# Task-Specific Expertise – Fine-tuning excels at specialized tasks or formats (making the model an expert in a specific area), while RAG keeps the model up-to-date with the latest external knowledge.

# Better Adaptability – A fine-tuned model can still give useful answers even if the retrieval component fails or returns incomplete information. Meanwhile, RAG ensures the system stays current without requiring you to retrain the model for every new piece of data.

# Efficiency – Fine-tuning provides a strong foundational knowledge base within the model, and RAG handles dynamic or quickly-changing details without the need for exhaustive re-training from scratch. This balance yields an efficient workflow and reduces overall compute costs.

# LoRA vs. QLoRA: Which One to Use?

# LoRA (Low-Rank Adaptation) – Fine-tunes only a small set of additional “adapter” weight matrices (in 16-bit precision), while leaving most of the original model unchanged. This significantly reduces the number of parameters that need updating during training.


# QLoRA (Quantized LoRA) :Combines LoRA with 4-bit quantization of the model weights, enabling efficient fine-tuning of very large models on minimal hardware. By using 4-bit precision where possible, it dramatically lowers memory usage and compute overhead.


# A simplified understanding of LORA and QLORA :


 # LLM Fine-Tuning Shortcuts: LoRA vs. QLoRA
 # 📘 LoRA (Low-Rank Adaptation) The Core Concept: Instead of updating billions of parameters in the original model, LoRA freezes the base model and trains a small set of external "adapter" matrices.
# The Analogy: Leaving a 1,000-page textbook completely untouched and only writing specialized updates on Post-it notes stuck to the pages.
# How it Alters Behaviour: It acts like a detour in the AI's thinking path. The input signal passes through the original model, gets intercepted by the adapter to apply the new style/rules, and merges back into the main output.
# Key Benefit: Saves Training Memory. Drastically reduces the number of parameters that need updating, making training much faster and cheaper.

# 🔬 QLoRA (Quantized LoRA) The Core Concept: Takes LoRA a step further by heavily compressing the frozen base model down to 4-bit precision using a specialized mathematical format (NF4), while keeping the LoRA adapters in high-resolution (16-bit).
# The Analogy: Shrinking that 1,000-page textbook into a blurry, micro-sized booklet to save massive amounts of storage space, then using high-definition Post-it notes on top.
# How it Alters Behaviour: The signal passes through the compressed 4-bit model to get the general logic. When it hits the high-resolution 16-bit adapter, the adapter not only applies the new data but automatically corrects the errors caused by the compression.
#  Key Benefit: Saves Storage & Hardware Memory. Allows massive enterprise-grade models to be fine-tuned on cheap, consumer-level hardware (like a single gaming GPU) with zero loss in final accuracy.

# Choosing the right Model is important + Method:

# To start with fine-tunning we need to decide between normal fine tunning, Reinforcement Learning,QLORA and LORA training:

# Reinforcement Learning(RL): is used when you need a model to excel at a specific behavior (e.g., tool-calling) using an environment and reward function rather than labeled data. We have several notebook examples, but for most use-cases, standard SFT is sufficient

# LoRA is a parameter efficient training method that typically keeps the base model’s weights frozen and trains a small set of added low-rank adapter weights (in 16-bit precision).

# QLoRA combines LoRA with 4-bit precision to handle very large models with minimal resources.

# Unsloth also supports full fine-tuning (FFT) and pretraining, which require significantly more resources, but FFT is usually unnecessary. When done correctly, LoRA can match FFT.


# Steps to Fine Tuning is 
# When preparing for fine-tuning, one of the first decisions you'll face is whether to use an instruct model or a base model.

# Lets understand the basic differences between a base model and an instruct model

# Base Models
# A base model has only undergone pre-training, it has read vast portions of the internet and learned onl one fundamental skill: predicting the next word in a sequence.

# how it behaves :It does not know how to follow directions or hold a conversations, if you prompt it with :"What is the capital of France ?",it might not answer the question. instead,
# it might output a list of other questions : What is the capital of Germany ? What is the capital of Italy ? because that is how the textbook or a quiz format looks on the internet.

# Best used for : Deep domain adaptation, if you want to train an AI on a completely new language, a massive library of proprietary medical textbooks, or a unique coding syntax it has never seen before, you start with a base model

# Instruct Models (The Conversationalist)
# An instruct model is a model that has already gone through instruction tuning and RLHf(Reinfrocement Learning from Human Feedback )by companies like Meta or Google

# • How it behaves: It has been trained specifically to understand the roles of "User" and "Assistant." It knows that when a human asks a question, it is supposed to provide a helpful, direct answer.
# • Best used for: Changing the tone, style, or output format of a conversational AI. If you want an AI that behaves exactly like ChatGPT but speaks strictly like a 1920s detective, or always outputs its answers in a highly structured, valid JSON format, you start with an instruct model.

# A case for Dataset

# Less than 300 rows: Choose Instruct
# The logic: A small dataset is not enough text to teach a model how to behave, it is only enough to show it a few examples of a specific style.
# An instruct model already has a massive head start—it already knows how to follow formatting, answer questions, and behave like a chatbot. Your small dataset simply steers its existing behavior. If you tried to use a base model with fewer than 300 rows, it would never learn how to hold a proper conversation, and it would just spit out chaotic text.

# 300–1,000 Rows: The "Grey Area" (Both Viable)
# The Logic: This is the middle ground where your data has a decent amount of influence.
# The Choice: If your 500 rows are perfectly formatted, high-quality Q&As, an instruct model will refine its tone brilliantly. If those 500 rows are raw, highly specialized code files or unique formatting patterns, a base model can start to pick up the pattern without being fighting its pre-existing conversational programming.

# 1,000+ Rows: Choose Base
# The Logic: With over 1,000 examples, you have enough raw data to actually "override" a model's foundational habits and teach it a completely new structure or massive domain of knowledge.
# Why it works: Because instruct models have been heavily aligned by tech companies, they can sometimes exhibit "rigidity" (they try too hard to be a helpful assistant, even when your training data is trying to make them do something else). Starting from a base model with a large dataset gives you a clean canvas. The model will smoothly adapt to your data's exact structure without any corporate guardrails or pre-trained conversational habits fighting back.


# What is Unsloth

# Unsloth makes finetuning LLMs like LLama-3 , Mistral, Phi-3 and Gemma


## Learnings

## 2026-10-15: Model evaluation

**Two kinds of evals**
- Model benchmarks (MMLU, ARC, HellaSwag) compare models. Product evals test my own system on my own data, like the FinSight golden set.
- Component evals grade parts of a pipeline separately: retriever (context precision and recall) vs generator (faithfulness, answer relevancy).

**Ran my first benchmark with lm-evaluation-harness**

```bash
uv pip install lm-eval torch transformers accelerate
lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-0.5B-Instruct \
  --tasks arc_easy --limit 200 --batch_size 4 --device cpu
```

- `--tasks arc_easy` pulls the `allenai/ai2_arc` dataset (grade-school science multiple choice) from Hugging Face automatically.
- Scoring: the model's likelihood of each of the four answers is compared, and the highest one counts as its answer.
- **Result:** acc 0.565 ± 0.035 (200 questions, zero-shot, CPU, about 1.5 minutes). Random guessing is 0.25.
- `--limit` is for testing only. Drop it for a real number, because the error margin shrinks.

**Problems I hit**
- `ModuleNotFoundError: No module named 'accelerate'`: the Hugging Face backend needs it, so `uv pip install accelerate`.
- Backslash line continuations must be the last character on the line, or put the command on one line.

**Key ideas**
- Each model size is trained separately, so a 0.5B result says nothing reliable about the 7B. Re-run evals per model.
- Benchmarks shortlist models. My own eval set decides, plus cost, latency and output reliability.
- Differences smaller than the standard error mean nothing.

**Plan for the PII fine-tune**
1. Hold out a test slice of the dataset before training and never train on it.
2. Run baselines on it: zero-shot Qwen and GPT-4o-mini.
3. After training, score the held-out slice with entity-level F1 per PII type and leak rate.
4. Run `lm_eval` on base vs base plus adapter (`peft=path/to/adapter`) as a check for catastrophic forgetting. A drop beyond the error margin is a warning.

**Still to try**
- Full `arc_easy` run without `--limit`.
- Same task on Qwen2.5-1.5B-Instruct to see the effect of size.
- A second task (`hellaswag`).