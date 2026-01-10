# Recursive Language Models (RLM) - Deep Dive Analysis

## The Problem: Context Length Limitations

Traditional LLMs face fundamental constraints:
- **Fixed context windows**: Even with extended context (100K, 1M tokens), there's a hard limit
- **Context rot**: Performance degrades with very long contexts - models struggle to attend to all relevant information
- **Cost scaling**: Longer contexts = higher API costs and latency
- **All-or-nothing loading**: Must load entire context upfront, even if only parts are relevant

## The RLM Solution: Recursive Self-Querying

RLMs transform the LM from a **pure function** (text → text) into a **recursive agent** that can:
1. **Decompose** complex queries into sub-problems
2. **Selectively access** parts of context as needed
3. **Recursively reason** through nested sub-queries
4. **Scale** to near-infinite context by treating it as external memory

---

## Architecture Breakdown: The Diagram Explained

### Level 0: Root RLM Call

```
┌─────────────────────────────────────────┐
│  INPUT: query + context                 │
│         ↓                                │
│  ┌───────────────┐                       │
│  │ Language Model│ → final response      │
│  └───────┬───────┘                       │
│          ↓                                │
│  ┌──────────────────┐                    │
│  │ Environment E    │                    │
│  │ (REPL)           │                    │
│  │ - context stored │                    │
│  │ - call_llm()     │                    │
│  └──────────────────┘                    │
└─────────────────────────────────────────┘
```

**Key Insight**: Instead of the context being in the prompt, it's stored in an **environment variable** accessible via a REPL.

### The REPL Environment

The Environment E is a Python REPL that provides:

1. **Context as a variable**: `context.read()` - programmatic access to the full context
2. **LM sub-call function**: `call_llm(sub_query, sub_context)` - ability to spawn recursive queries
3. **Execution sandbox**: Code runs in isolated namespace with injected functions

**Example REPL interaction**:
```python
# LM can generate and execute this code:
relevant_section = context.read()[:1000]  # Get first 1000 chars
sub_result = call_llm(
    query="Summarize this section",
    context=relevant_section
)
print(sub_result)
```

### Level 1: Recursive Sub-Calls

When the root LM executes `call_llm()`, it spawns a new RLM instance at depth=1:

```
┌─── RLM (depth=1) Left ────┐      ┌─── RLM (depth=1) Right ───┐
│ INPUT: sub-query 1         │      │ INPUT: sub-query 2         │
│        sub-context 1       │      │        sub-context 2       │
│         ↓                  │      │         ↓                  │
│ ┌───────────────┐          │      │ ┌───────────────┐          │
│ │Language Model │          │      │ │Language Model │          │
│ └───────┬───────┘          │      │ └───────┬───────┘          │
│         ↓                  │      │         ↓                  │
│   sub-response 1 ──────────┼──────┼─→ ┌──────────────────┐    │
└────────────────────────────┘      │   │ Environment E    │    │
                                    │   │ (REPL)           │    │
                                    │   └─────┬─┬─┬────────┘    │
                                    │         │ │ │             │
                                    │       depth=2, 3, ...     │
                                    └──────────────────────────┘
```

**Critical Details**:
- Each sub-call gets its **own sub-context** (not the full original context)
- Sub-queries are **specialized** - they solve specific sub-problems
- Sub-responses **propagate back up** to the parent call
- The right diagram shows **further recursion** - depth=2, 3, etc.

### Recursive Depth Control

**Depth=1 (Basic RLM)**:
- Root LM can call `Sub_RLM`
- Sub calls **cannot** make further recursive calls
- Good for simple decomposition

**Depth=∞ (Full Recursion)**:
- Replace `Sub_RLM` with `RLM_REPL`
- Each sub-call can make its own sub-calls
- Enables hierarchical problem decomposition
- Like a tree of reasoning steps

---

## Control Flow: Step-by-Step Execution

### 1. **Initialization**
```python
rlm = RLM_REPL()
response = rlm.completion(query="Analyze document X", context=huge_document)
```

### 2. **Root LM Receives Prompt**
The LM gets a prompt like:
```
You have access to a Python REPL with:
- context: The full document stored as a variable
- call_llm(query, context): Make sub-queries to analyze specific parts

Query: Analyze document X

Generate Python code to solve this.
```

### 3. **LM Generates Code with Recursive Calls**
```python
# LM's generated code:

# Step 1: Check document structure
structure = context.metadata()

# Step 2: Extract sections
intro = context.read(0, 1000)
methods = context.read(5000, 8000)
results = context.read(10000, 15000)

# Step 3: Analyze each section recursively
intro_summary = call_llm(
    "Summarize the introduction",
    intro
)

methods_summary = call_llm(
    "What methodology is used?",
    methods
)

results_summary = call_llm(
    "What are the key findings?",
    results
)

# Step 4: Synthesize
final_analysis = f"""
Introduction: {intro_summary}
Methods: {methods_summary}
Results: {results_summary}
"""

print(final_analysis)
```

### 4. **REPL Executes Code**
- Runs line by line
- When it hits `call_llm()`, spawns a new LM instance
- Waits for sub-response
- Continues execution with the result

### 5. **Sub-LM Execution**
Each `call_llm()` creates a **new, independent LM call**:
- Gets only the **sub-context** (e.g., just the intro section)
- Processes the **sub-query** (e.g., "Summarize the introduction")
- Returns **sub-response** (e.g., "This paper introduces...")
- If depth > 1, can make its own recursive calls

### 6. **Response Aggregation**
- Sub-responses populate variables in the REPL
- Final `print()` statement captures the output
- Root LM returns this as the final response

---

## Why This Works: The Key Innovations

### 1. **Selective Context Access**
Traditional: Load all 10M tokens → LM tries to attend to everything
RLM: Load 10M tokens into variable → LM reads only what it needs (maybe 50K tokens total across all recursive calls)

### 2. **Hierarchical Decomposition**
```
Complex Query
├─ Sub-query 1 (handles part A)
├─ Sub-query 2 (handles part B)
│  ├─ Sub-sub-query 2.1
│  └─ Sub-sub-query 2.2
└─ Sub-query 3 (synthesizes A + B)
```

Each level works with **manageable context**, avoiding context rot.

### 3. **Lazy Loading**
Don't process what you don't need:
- Traditional: All context in prompt always
- RLM: Only load context chunks when `call_llm()` is invoked

### 4. **Cost Efficiency**
- Smaller, focused queries → can use cheaper models (GPT-4-mini vs GPT-4)
- Only pay for tokens actually processed
- Parallel sub-queries possible

---

## Concrete Example: Long Document Q&A

**Scenario**: Answer questions about a 100-page PDF (200K tokens)

### Traditional Approach
```
Prompt: [200K token PDF] + "What are the main conclusions?"
→ Single GPT-4 call with full context
→ Expensive, slow, may miss details in middle of document
```

### RLM Approach
```python
# Root LM strategy:
table_of_contents = context.read(0, 500)  # Get TOC

# Find conclusion section location
conclusion_location = call_llm(
    "Where is the conclusion section?",
    table_of_contents
)

# Read just that section
conclusion_text = context.read(conclusion_location.start, conclusion_location.end)

# Analyze it
main_conclusions = call_llm(
    "What are the main conclusions?",
    conclusion_text
)

print(main_conclusions)
```

**Result**:
- Only processed ~5K tokens total instead of 200K
- Faster, cheaper, more accurate (focused attention)
- Can use GPT-4-mini instead of GPT-4

---

## Implementation Details

### RLM_REPL Class Structure
```python
class RLM_REPL:
    def completion(self, query, context):
        # 1. Create REPL environment
        repl_env = REPL()
        repl_env.inject("context", context)
        repl_env.inject("call_llm", self.sub_llm_call)

        # 2. Prompt LM to generate code
        prompt = f"""
        You have a REPL with:
        - context variable
        - call_llm(query, ctx) function

        Query: {query}
        Generate Python code to answer this.
        """

        code = llm.generate(prompt)

        # 3. Execute code in REPL
        output = repl_env.execute(code)

        return output

    def sub_llm_call(self, sub_query, sub_context):
        # Spawn new RLM instance (depth+1)
        sub_rlm = Sub_RLM()  # or RLM_REPL for infinite depth
        return sub_rlm.completion(sub_query, sub_context)
```

### Sub_RLM (Depth=1 Only)
```python
class Sub_RLM:
    def completion(self, query, context):
        # Simple LM call, no REPL
        prompt = f"Context: {context}\n\nQuery: {query}"
        return llm.generate(prompt)
```

---

## Comparison with Other Approaches

### vs. RAG (Retrieval-Augmented Generation)
- **RAG**: Embed context → retrieve top-k chunks → LM processes chunks
- **RLM**: LM decides what to retrieve and how to process it
- **Advantage**: RLM has more control and can do multi-hop reasoning

### vs. ReAct (Reasoning + Acting)
- **ReAct**: LM generates actions → Execute → LM sees results → Repeat
- **RLM**: Similar, but actions include **recursive LM calls**
- **Advantage**: Can delegate sub-reasoning, not just tool use

### vs. Test-Time Compute (o1 style)
- **o1**: Internal chain-of-thought, opaque reasoning
- **RLM**: Explicit recursive structure, debuggable
- **Advantage**: More controllable, can see exactly how it decomposes problems

---

## Performance Results

From the paper:
- **GPT-4-mini + RLM > GPT-4** on OOLONG benchmark (long-context)
- **2x more correct answers** than standard approach
- **Cheaper per query** (uses smaller model recursively)
- **No degradation at 10M+ tokens** (because it doesn't load all tokens)

---

## Key Takeaways

1. **Paradigm Shift**: LMs as recursive reasoners, not just text completors
2. **Unbounded Context**: Treat context as external memory, not prompt stuffing
3. **Hierarchical Reasoning**: Decompose → Solve → Synthesize
4. **Efficiency**: Smaller models + selective processing = better than big models with full context
5. **Future Direction**: This is likely the next evolution after CoT and ReAct

## The Big Picture

RLMs represent a fundamental shift:
- **Before**: LM(context + query) → response
- **After**: LM(query) + access_to(context, recursive_self) → response

This mirrors how humans solve complex problems:
- We don't memorize entire documents
- We skim, identify relevant sections, deep-dive where needed
- We break problems into sub-problems
- We synthesize insights hierarchically

RLMs make LMs work more like human researchers, not just pattern-matching text completors.
