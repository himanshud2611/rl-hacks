# On-Policy Distillation

**Source**: [Thinking Machines Blog](https://thinkingmachines.ai/blog/on-policy-distillation/)
**Date**: December 23, 2025

## Overview

On-policy distillation is a hybrid post-training approach that combines RL's on-policy sampling with dense token-level supervision from a teacher model. It trains the student model on its own outputs while using a teacher to provide detailed feedback on each token.

## Key Concepts

### The Problem
- **RL**: On-policy but sparse feedback (one reward per episode)
- **SFT**: Dense feedback but off-policy (trains on teacher examples, not student's actual outputs)
- **Goal**: Get dense feedback while staying on-policy

### The Solution
1. Sample trajectories from the **student model** (on-policy)
2. Compute **teacher log probabilities** on those trajectories
3. Use **reverse KL divergence** as the loss function
4. Train via policy gradient with teacher feedback as advantages

### Technical Details
- **Loss**: Reverse KL between student and teacher distributions per token
- **Advantage**: Mode-seeking behavior (learns specific strategies, not averaging)
- **Efficiency**: Only requires forward passes from teacher (no gradients)

## Key Results

### Math Reasoning (AIME'24 with Qwen3-8B)
| Method | GPU Hours | Accuracy |
|--------|-----------|----------|
| Off-policy distillation | - | 60% |
| RL | 17,920 | 67.6% |
| **On-policy distillation** | **1,800** | **74.4%** |

**Cost reduction**: 9-30x versus alternatives
**Sample efficiency**: 7-10x fewer gradient steps than RL (50-100x compute savings)

### Continual Learning
- Fine-tuned Qwen3-8B on company documents (lost instruction-following)
- Recovered capabilities via distillation from original model as teacher
- Maintained both new knowledge and original capabilities

## Advantages

1. **Dense Supervision**: O(N) bits per episode vs O(1) for RL
2. **Data Reusability**: Can train multiple epochs on same prompts
3. **On-Policy Consistency**: Avoids compounding errors from off-policy training
4. **Compute Efficient**: Parallelizable teacher computation without backprop
5. **Continual Learning**: Can alternate between learning and recovery

## Limitations

- Requires a strong teacher model
- Student needs sufficient foundation knowledge
- Doesn't replace RL for exploring novel solution spaces
- Effectiveness depends on teacher quality

## Key Insight

> RL searches through semantic strategy space; distillation shortcuts this by directly teaching the discovered strategy without modeling exploration phases.

---

## Deep Dive: Conceptual Understanding

### The Fundamental Trade-off in Post-Training

Post-training methods face a critical distribution mismatch problem:

**Off-Policy Training (Traditional SFT/Distillation)**
- You collect examples from a teacher model: "Given prompt X, teacher outputs Y"
- Student learns to mimic these teacher trajectories
- **Problem**: Student makes different mistakes than teacher. When student starts generating its own text, it creates tokens the teacher would never produce
- This leads to **compounding errors**: student ventures into states it never saw during training, makes worse mistakes, ventures further off-distribution
- Like learning to drive by watching videos - you never learn to recover from your own mistakes

**On-Policy Training (Reinforcement Learning)**
- Student generates its own outputs and gets feedback on them
- Learns to recover from its own mistakes since it's training on its own distribution
- **Problem**: Feedback is sparse - one reward at the end of an entire sequence (could be hundreds of tokens)
- Which of the 200 tokens were good? Which were bad? Hard to assign credit
- Requires extensive exploration and many samples to learn effectively
- Expensive: needs 17,920 GPU hours in their example

### The On-Policy Distillation Innovation

**The Core Idea**: What if we could get token-level feedback (dense supervision) while training on the student's own outputs (on-policy)?

**How It Works - Step by Step**:

1. **Generate from student**: Student model generates a completion for a prompt
   - E.g., solving a math problem, the student produces its own reasoning chain

2. **Get teacher's opinion on each token**: For the exact sequence the student generated, ask the teacher "what probability would YOU have assigned to each token?"
   - Not asking teacher to generate its own solution
   - Asking teacher to evaluate the student's solution token-by-token

3. **Compute divergence**: Measure how different the student's token probabilities are from the teacher's
   - Using reverse KL: forces student to put probability mass where teacher does (mode-seeking)
   - This is different from forward KL which would spread student's probability across all teacher modes

4. **Update student**: Use this per-token divergence as the learning signal
   - Dense feedback: O(N) bits of information per sequence (one per token)
   - On-policy: training on sequences the student actually generates

### Why Reverse KL Matters

**Forward KL** (traditional distillation): `KL(Teacher || Student)`
- Student tries to cover all modes where teacher has probability mass
- "Mean-seeking" behavior - student spreads out to cover everything teacher might do
- Good for coverage, but student might generate things teacher rarely does

**Reverse KL** (on-policy distillation): `KL(Student || Teacher)`
- Student tries to put mass only where teacher has high probability
- "Mode-seeking" behavior - student focuses on what teacher actually does
- When student generates a token, teacher provides strong signal if it's on the right track
- Reduces exposure bias: student trained on its own distribution, not teacher's

### The Information Efficiency Advantage

**Why it's 50-100x more compute efficient than RL**:

1. **Dense signal**: Every token gets feedback vs one reward per sequence
   - 200 tokens → 200 learning signals instead of 1

2. **No wasted exploration**: Don't need to explore bad strategies extensively
   - Teacher already knows good strategies, directly teaches them
   - RL must try many bad approaches to discover good ones

3. **Data reusability**: Can train multiple epochs on same prompts
   - Because learning distributions, not memorizing specific answers
   - RL with sparse rewards quickly overfits to specific trajectories

4. **Computational efficiency**: Only need teacher forward passes
   - Can batch and parallelize teacher inference
   - No need for teacher gradients or backpropagation
   - Can use quantized/optimized teacher inference

### The Continual Learning Connection

When you fine-tune a model on new data (e.g., company documents), it often suffers **catastrophic forgetting** - loses its general capabilities:

**Traditional approach**: Mix old and new data (expensive, may not have old data)

**On-policy distillation approach**:
1. Fine-tune on new domain (model becomes domain-specialized but loses general skills)
2. Use original model as teacher
3. Student generates outputs, original model provides feedback
4. Student learns to satisfy both: new domain knowledge + original capabilities

This works because:
- Student maintains its new knowledge (parameters already updated)
- Teacher guides student back toward original behaviors where needed
- On-policy ensures student learns in contexts it actually encounters

### When It Works vs When It Doesn't

**Works Best When**:
- Teacher is significantly stronger than student
- Task requires following known good strategies (math, coding, instruction-following)
- Student has foundation knowledge, just needs refinement
- You have limited compute budget

**Doesn't Work When**:
- No strong teacher exists (truly novel domains)
- Need to explore fundamentally new strategies teacher doesn't know
- Teacher-student gap is too large (student can't even approximate teacher's distribution)
- Task requires creativity/diversity rather than correctness

### The Meta-Insight

On-policy distillation reveals something important about learning efficiency:

**Dense supervision >> Sparse rewards** when learning known strategies

The order of magnitude improvement (50-100x) suggests that most of RL's compute goes into:
1. Exploring strategy space (trying bad approaches)
2. Assigning credit across long sequences (which tokens mattered?)

If you have a teacher that already knows good strategies, you can skip both:
1. Teacher shows you what good strategies look like (no exploration needed)
2. Teacher provides token-level feedback (no credit assignment needed)

This is why distillation is so much more efficient - it's not just "slightly better", it's fundamentally avoiding expensive phases of learning.

### Practical Mental Model

Think of it like learning a skill:

**RL**: Trial and error. Try many approaches, get sparse feedback ("good" or "bad" at the end). Eventually discover what works through extensive practice.

**Off-policy distillation**: Watch expert demonstrations. Try to copy exactly what expert does. Problem: when you make a mistake, you're in unfamiliar territory.

**On-policy distillation**: You perform the skill, expert watches and gives you detailed feedback at each step. "That motion was good, this one was wrong, here's what I would have done." You learn to recover from your own mistakes with expert guidance.

The last approach combines the best of both: you're learning from your own experience (on-policy) but with expert guidance at every step (dense supervision).

---

## Applications

1. **Cost-effective post-training**: Achieve frontier performance with limited compute
2. **Personalization**: Adapt models to specific domains without catastrophic forgetting
3. **Continual learning**: Maintain general capabilities while learning new tasks
4. **Knowledge recovery**: Restore lost capabilities after domain-specific fine-tuning

## Implementation

- Available in open-source **Tinker training framework**
- Code in cookbook repository

---

## Agenda for Further Study

1. **Understand reverse KL divergence**
   - Why mode-seeking vs mean-seeking?
   - How does it reduce exposure bias?

2. **Compare with other distillation methods**
   - Traditional knowledge distillation
   - Off-policy distillation
   - Online distillation

3. **Explore implementation details**
   - Policy gradient formulation
   - Advantage computation from teacher logprobs
   - Training stability techniques

4. **Investigate applications**
   - Domain adaptation use cases
   - Continual learning strategies
   - Multi-task learning scenarios

5. **Study limitations and failure modes**
   - When does it underperform pure RL?
   - Teacher-student capability gap requirements
   - Distribution shift handling

6. **Experiment with variations**
   - Different divergence measures
   - Mixed on/off-policy approaches
   - Dynamic teacher-student ratios
