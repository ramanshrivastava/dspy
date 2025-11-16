# 🎓 Mini-DSPy: Historically-Grounded Learning Framework

**Build DSPy from Scratch with Deep Understanding**

---

## 📖 Overview

This framework guides you through implementing DSPy from scratch using a reasoning-based, historically-grounded approach. Each phase maps to DSPy's evolution and includes Architecture Decision Records (ADRs), learning checkpoints, and comparative analysis.

**Target**: 3,000-5,000 lines of Python (vs 15,000+ in real DSPy)
**Focus**: Core abstractions, not optimization
**Method**: Incremental, reasoning-driven commits

---

## 🗺️ Historical Context: DSPy Evolution

### **Timeline: From DSP to DSPy**

| Date | Version | Milestone | Paper/Event | Significance |
|------|---------|-----------|-------------|--------------|
| **Dec 2022** | DSP v0.1 | Demonstrate-Search-Predict | [DSP Paper](https://arxiv.org/abs/2212.14024) | Original RAG pipeline framework |
| **Jan 2023** | DSP v0.2 | Introducing DSP Compiler | Twitter announcement | First optimization attempts |
| **Aug 2023** | DSPy v1.0 | Complete rewrite as DSPy | Framework announcement | Declarative programming model |
| **Oct 2023** | DSPy v2.0 | Research paper published | [DSPy Paper](https://arxiv.org/abs/2310.03714) | Academic foundation |
| **Dec 2023** | DSPy v2.1 | Assertions added | [Assertions Paper](https://arxiv.org/abs/2312.13382) | Computational constraints |
| **Jun 2024** | DSPy v2.4 | MIPROv2 optimizer | [MIPRO Paper](https://arxiv.org/abs/2406.11695) | Multi-stage optimization |
| **Oct 2024** | DSPy v2.5 | Pydantic v2 migration | Major refactor | Type-safe signatures |
| **Nov 2024** | DSPy v3.0 | Adapter pattern | Architecture refactor | Pluggable formatters |
| **Jul 2025** | DSPy v3.0.4 | GEPA optimizer | [GEPA Paper](https://arxiv.org/abs/2507.19457) | Reflective optimization |

### **Key Design Influences**

1. **PyTorch** (2016): Composition-based module system → `dspy.Module`
2. **scikit-learn** (2011): `fit()` pattern → `compile()` in DSPy
3. **Keras** (2015): Declarative layer specification → `Signature`
4. **AutoML** (2010s): Hyperparameter optimization → Teleprompters
5. **JAX** (2018): Functional transformations → Stateless compilation

---

## 🏗️ Architecture Overview

```
Python Code (User Program)
         ↓
    MODULE SYSTEM       (Phase 1: ~400 lines)
    - Module base class
    - Parameter tracking
    - Composition
         ↓
    SIGNATURES          (Phase 2: ~600 lines)
    - Pydantic-based specs
    - Field definitions
    - Type validation
         ↓
    PREDICT             (Phase 3: ~500 lines)
    - Core LM caller
    - Demo management
    - Callback system
         ↓
    ADAPTERS            (Phase 4: ~400 lines)
    - Format prompts
    - Parse responses
    - Function calling
         ↓
    LM CLIENTS          (Phase 5: ~600 lines)
    - OpenAI/Anthropic
    - Caching
    - Usage tracking
         ↓
    OPTIMIZERS          (Phase 6: ~800 lines)
    - BootstrapFewShot
    - MIPROv2 (simple)
    - Evaluation
         ↓
    Results
```

---

## 📊 Phased Implementation Plan

### **Phase 1: Module System** (4 commits, ~400 lines)

**Historical Context**: Python module pattern established, PyTorch-style composition (2016)

| Commit | Feature | ADR | DSPy Ref | Lines | Learning Focus |
|--------|---------|-----|----------|-------|----------------|
| 1.1 | BaseModule + save/load | ADR-001 | `primitives/base_module.py` | 150 | Serialization, deep copy |
| 1.2 | Module + metaclass | ADR-002 | `primitives/module.py:18-38` | 100 | Metaclass pattern, initialization |
| 1.3 | Parameter tracking | ADR-003 | `primitives/module.py:103-106` | 100 | Tree traversal, named_parameters |
| 1.4 | Callbacks & history | ADR-004 | `primitives/module.py:65-82` | 50 | Observer pattern, execution tracking |

**Checkpoint 1**: Implement a custom Module that tracks its predictors

---

### **Phase 2: Signatures** (5 commits, ~600 lines)

**Historical Context**: Pydantic v2 migration (Oct 2024), type-safe programming

| Commit | Feature | ADR | DSPy Ref | Lines | Learning Focus |
|--------|---------|-----|----------|-------|----------------|
| 2.1 | Field definitions | ADR-005 | `signatures/field.py` | 80 | Pydantic Field patterns |
| 2.2 | Base Signature class | ADR-006 | `signatures/signature.py:40-50` | 150 | BaseModel inheritance |
| 2.3 | SignatureMeta (type creation) | ADR-007 | `signatures/signature.py:40-136` | 200 | Metaclass magic, frame introspection |
| 2.4 | String signatures | ADR-008 | `signatures/utils.py:make_signature` | 100 | DSL parsing, dynamic class creation |
| 2.5 | Signature manipulation | ADR-009 | `signatures/signature.py:prepend/append` | 70 | Immutable updates, method chaining |

**Checkpoint 2**: Create signatures 3 ways, manipulate fields dynamically

---

### **Phase 3: Predict Module** (4 commits, ~500 lines)

**Historical Context**: Core abstraction from DSP v0.1 (2022)

| Commit | Feature | ADR | DSPy Ref | Lines | Learning Focus |
|--------|---------|-----|----------|-------|----------------|
| 3.1 | Predict base + forward | ADR-010 | `predict/predict.py:19-40` | 150 | Module + Parameter inheritance |
| 3.2 | Demo management | ADR-011 | `predict/predict.py:41-66` | 100 | Few-shot example storage |
| 3.3 | State serialization | ADR-012 | `predict/predict.py:47-89` | 150 | Save/load predictors |
| 3.4 | ChainOfThought | ADR-013 | `predict/chain_of_thought.py` | 100 | Signature extension pattern |

**Checkpoint 3**: Build a CoT predictor, save and load it

---

### **Phase 4: Adapters** (4 commits, ~400 lines)

**Historical Context**: DSPy v3.0 refactor (Nov 2024) - adapter pattern introduced

| Commit | Feature | ADR | DSPy Ref | Lines | Learning Focus |
|--------|---------|-----|----------|-------|----------------|
| 4.1 | Adapter base class | ADR-014 | `adapters/base.py:22-65` | 100 | Abstract base, format/parse |
| 4.2 | ChatAdapter formatting | ADR-015 | `adapters/chat_adapter.py:format` | 150 | Prompt construction, demos |
| 4.3 | ChatAdapter parsing | ADR-016 | `adapters/chat_adapter.py:parse` | 100 | Response extraction |
| 4.4 | Callback integration | ADR-017 | `adapters/base.py:60-65` | 50 | Decorator pattern (@with_callbacks) |

**Checkpoint 4**: Implement a custom adapter for JSON output

---

### **Phase 5: LM Clients** (5 commits, ~600 lines)

**Historical Context**: LiteLLM integration (2023), multi-provider support

| Commit | Feature | ADR | DSPy Ref | Lines | Learning Focus |
|--------|---------|-----|----------|-------|----------------|
| 5.1 | BaseLM interface | ADR-018 | `clients/base_lm.py` | 100 | Abstract LM interface |
| 5.2 | LM class + LiteLLM | ADR-019 | `clients/lm.py:__init__` | 150 | Provider abstraction |
| 5.3 | LM.forward() + caching | ADR-020 | `clients/lm.py:forward` | 200 | API calls, disk cache |
| 5.4 | Usage tracking | ADR-021 | `utils/usage_tracker.py` | 100 | Token counting, cost calculation |
| 5.5 | Settings & context | ADR-022 | `dsp/utils/settings.py` | 50 | Thread-local configuration |

**Checkpoint 5**: Call OpenAI API, track usage, implement caching

---

### **Phase 6: Optimizers** (6 commits, ~800 lines)

**Historical Context**: BootstrapFewShot (DSP 2022), MIPROv2 (Jun 2024)

| Commit | Feature | ADR | DSPy Ref | Lines | Learning Focus |
|--------|---------|-----|----------|-------|----------------|
| 6.1 | Teleprompter base | ADR-023 | `teleprompt/teleprompt.py` | 50 | Abstract compile() method |
| 6.2 | Example & Prediction | ADR-024 | `primitives/example.py` | 150 | Data containers, with_inputs() |
| 6.3 | Evaluate framework | ADR-025 | `evaluate/evaluate.py` | 200 | Parallel evaluation, metrics |
| 6.4 | BootstrapFewShot setup | ADR-026 | `teleprompt/bootstrap.py:36-100` | 150 | Teacher/student pattern |
| 6.5 | Bootstrap execution | ADR-027 | `teleprompt/bootstrap.py:_bootstrap` | 200 | Demo collection, metric validation |
| 6.6 | LabeledFewShot (simple) | ADR-028 | `teleprompt/vanilla.py` | 50 | Labeled demo selection |

**Checkpoint 6**: Optimize a QA program with BootstrapFewShot

---

### **Phase 7: Integration** (3 commits, ~500 lines)

**Historical Context**: DSPy as production framework (2024+)

| Commit | Feature | ADR | DSPy Ref | Lines | Learning Focus |
|--------|---------|-----|----------|-------|----------------|
| 7.1 | Public API (__init__.py) | ADR-029 | `dspy/__init__.py` | 50 | Export management |
| 7.2 | Retrieve module | ADR-030 | `retrievers/retrieve.py` | 200 | Retrieval abstraction |
| 7.3 | End-to-end RAG | ADR-031 | `tests/examples/test_baleen.py` | 250 | Complete pipeline |

**Checkpoint 7**: Build and optimize a full RAG system

---

## 📝 Architecture Decision Record (ADR) Template

Each commit includes a detailed ADR following this structure:

```markdown
# ADR-XXX: [Decision Title]

**Status**: Accepted
**Date**: 2025-01-XX
**Commit**: [hash]
**DSPy Reference**: `[file:line]`
**Related Papers**: [Paper names/links]

---

## Context

**What problem are we solving?**
[Detailed problem description]

**What constraints exist?**
- Constraint 1
- Constraint 2

**What did DSPy need to handle?**
[Specific DSPy requirements]

---

## Decision

**What approach did we choose?**

```python
# Our mini-DSPy implementation
[code snippet showing the decision]
```

---

## Rationale

### Why This Approach?

1. **Reason 1**: [Detailed explanation]
2. **Reason 2**: [Detailed explanation]
3. **Reason 3**: [Detailed explanation]

### Alternatives Considered

#### Alternative A: [Name]
- **Description**: [What it is]
- **Pros**: [Benefits]
- **Cons**: [Drawbacks]
- **Why Rejected**: [Reason]

#### Alternative B: [Name]
- **Description**: [What it is]
- **Pros**: [Benefits]
- **Cons**: [Drawbacks]
- **Why Rejected**: [Reason]

---

## DSPy Comparison

### What Real DSPy Does

**File**: `dspy/[path]/[file].py` (lines X-Y)

```python
# Real DSPy implementation
[actual DSPy code snippet]
```

**Key differences**:
1. **Real DSPy**: [Feature/complexity]
   - **Mini-DSPy**: [Simplified version]
   - **Reason**: [Why we simplified]

2. **Real DSPy**: [Feature/complexity]
   - **Mini-DSPy**: [Simplified version]
   - **Reason**: [Why we simplified]

---

## Historical Evolution

### DSP (Dec 2022)
[How this feature was in original DSP]

### DSPy v1.0 (Aug 2023)
[How it changed in DSPy v1]

### DSPy v2.0 (Oct 2023)
[How it evolved in v2]

### DSPy v3.0 (Nov 2024)
[Modern approach]

**Evolution Summary**: [How and why the design changed]

---

## Trade-offs

### Benefits ✅
1. **Benefit 1**: [Explanation]
2. **Benefit 2**: [Explanation]
3. **Benefit 3**: [Explanation]

### Limitations ❌
1. **Limitation 1**: [What we can't do]
2. **Limitation 2**: [Performance cost]
3. **Limitation 3**: [Feature gap]

---

## Learning Outcomes

After implementing this commit, you should understand:

1. **[Concept 1]**: [What to learn]
   - Why this matters for LM programming
   - How DSPy uses this pattern elsewhere

2. **[Concept 2]**: [What to learn]
   - Connection to [related concept]
   - Practical implications

3. **[Concept 3]**: [What to learn]
   - Common pitfalls
   - Best practices

---

## References

### DSPy Codebase
- **Main file**: `/home/user/dspy/dspy/[path]/[file].py` (lines X-Y)
- **Related files**:
  - `[file1].py` - [purpose]
  - `[file2].py` - [purpose]

### Papers
- **Primary**: [Paper name] - [link]
  - Section X.Y explains [relevant part]
- **Related**: [Paper name] - [link]

### External Resources
- PyTorch pattern: [link to PyTorch docs]
- Pydantic docs: [link]
- Academic paper: [citation]

---

## Exercises

### Exercise 1: Understand the Code
**Task**: Explain how [specific feature] works
**Difficulty**: ⭐☆☆☆☆
**Time**: 15 minutes

**Questions**:
1. What happens when [scenario]?
2. Why did we use [pattern] instead of [alternative]?
3. How would you extend this to support [feature]?

### Exercise 2: Extend the Feature
**Task**: Add support for [new capability]
**Difficulty**: ⭐⭐⭐☆☆
**Time**: 45 minutes

**Steps**:
1. Modify [file] to add [feature]
2. Update [test] to verify behavior
3. Compare your solution to [reference]

**Hints**:
- Consider how DSPy handles this in `[file:line]`
- You'll need to understand [concept]

### Exercise 3: Debug the Bug
**Task**: We've introduced a subtle bug in [component]
**Difficulty**: ⭐⭐⭐⭐☆
**Time**: 60 minutes

**Bug**: [Description of what breaks]
**Symptoms**: [What you'll observe]
**Hint**: Check [specific area]

---

## Performance Considerations

### Complexity Analysis
- **Time**: O([complexity])
- **Space**: O([complexity])

### Real DSPy Optimizations
1. **Optimization 1**: [What DSPy does]
   - **Impact**: [Performance gain]
   - **Why we skip**: [Reason]

2. **Optimization 2**: [What DSPy does]
   - **Impact**: [Performance gain]
   - **Why we skip**: [Reason]

### Benchmark
```bash
python benchmarks/phase_X_bench.py
```

**Expected**: [What you should see]
**If slow**: [Common causes]

---

## Next Steps

**Before moving to the next commit**:
- [ ] Read the implementation carefully
- [ ] Run all tests (`pytest tests/phase_X/`)
- [ ] Complete at least 2 exercises
- [ ] Explain the design to yourself out loud
- [ ] Read the referenced DSPy code

**Ready for next commit when you can**:
- Explain the design decision without looking at notes
- Implement a variant of the feature
- Debug issues independently
```

---

## 🎯 Learning Checkpoint Template

After each phase, complete a comprehensive checkpoint:

```markdown
# Phase X Checkpoint: [Phase Name]

## Self-Assessment Quiz

### Conceptual Understanding

#### Question 1: Design Rationale
**Q**: Why did we choose [design decision] over [alternative]?

**Answer**: [Expected answer with reasoning]

**Reference**: ADR-XXX, DSPy code at `[file:line]`

**Explanation**: [Deeper dive into the reasoning]

#### Question 2: Trade-off Analysis
**Q**: What do we gain and lose with [approach]?

**Answer**:
- **Gain**: [Benefits]
- **Lose**: [Limitations]

#### Question 3: Historical Context
**Q**: How did this feature evolve from DSP to DSPy v3?

**Answer**: [Evolution timeline with key changes]

---

### Code Comprehension

#### Question 1: Code Reading
**Q**: What does this code do?
```python
[code snippet from implementation]
```

**Answer**: [Line-by-line explanation]

**DSPy Equivalent**: `[file:line]`

#### Question 2: Trace Execution
**Q**: Trace the execution of [method call]

**Answer**: [Step-by-step trace]

---

## Hands-On Exercises

### Exercise 1: Extend the Feature
**Task**: Add support for [new feature]
**Difficulty**: ⭐⭐☆☆☆
**Estimated Time**: 30 minutes
**Learning Goal**: Understand [concept]

**Requirements**:
1. [Requirement 1]
2. [Requirement 2]

**Hints**:
- DSPy does this in `[file:line]`
- You'll need to modify `[files]`
- Consider [edge case]

**Solution**: `solutions/phase_X_exercise_1.py`

---

### Exercise 2: Debug the Code
**Task**: We've introduced a bug. Find and fix it.
**Difficulty**: ⭐⭐⭐☆☆
**Estimated Time**: 45 minutes

**Bug Description**: [What breaks]
**Symptoms**: [Error message or wrong behavior]
**File**: `src/phase_X/[component].py`

**Debugging Steps**:
1. Run `pytest tests/phase_X/test_buggy.py`
2. Observe [symptom]
3. Hypothesis: [what might be wrong]
4. Fix and verify

**Solution**: `solutions/phase_X_bug_fix.py`

---

### Exercise 3: Build Something New
**Task**: Combine Phase X components to build [application]
**Difficulty**: ⭐⭐⭐⭐☆
**Estimated Time**: 90 minutes

**Application**: [Description]
**Requirements**: [Detailed spec]

**Example**: Build a multi-step reasoning system using:
- Module composition
- Custom signatures
- ChainOfThought predictor

---

## Comparative Analysis

### Mini-DSPy vs Real DSPy

| Aspect | Mini-DSPy | Real DSPy | Why Different? |
|--------|-----------|-----------|----------------|
| **Lines of code** | [X] | [Y] | [Reason] |
| **Features** | [List] | [List] | [Focus on core] |
| **Performance** | [Benchmark] | [Benchmark] | [Optimization skipped] |
| **Error handling** | [Simple] | [Comprehensive] | [Learning focus] |
| **Type system** | [Basic] | [Full Pydantic] | [Simplified] |

---

## Performance Benchmark

### Run Benchmark
```bash
python benchmarks/phase_X_benchmark.py
```

### Expected Results
```
Mini-DSPy: [metric]
Real DSPy: [metric]
Ratio: [X.XX]x slower
```

**Analysis**:
- **Bottleneck**: [Where is it slow?]
- **Why**: [Root cause]
- **DSPy's Solution**: [How they optimize]
- **Trade-off**: [What we sacrifice for simplicity]

---

## DSPy Code Reading Assignment

### Before Next Phase

**Required Reading**:
1. Read `dspy/[file].py` (lines X-Y)
   - Focus on [specific pattern]
   - Note how they handle [edge case]

2. Read `dspy/[file2].py` (lines A-B)
   - Compare to our implementation
   - Identify 3 differences

**Questions to Answer**:
1. How does DSPy handle [scenario]?
2. What optimization do they use in [section]?
3. Why did they structure [code] this way?

---

## Readiness Checklist

### Technical Understanding
- [ ] Can explain all design decisions without notes
- [ ] Completed all 3 exercises
- [ ] Read referenced DSPy code
- [ ] Answered all quiz questions correctly
- [ ] Can trace execution mentally

### Practical Skills
- [ ] Can modify code to add features
- [ ] Can debug issues independently
- [ ] Can explain trade-offs clearly
- [ ] Can compare mini vs real DSPy

### Conceptual Mastery
- [ ] Understand historical context
- [ ] Know why design evolved
- [ ] Can critique design decisions
- [ ] Can propose alternatives

---

## Next Steps

**You're ready for Phase X+1 when**:
1. All checkboxes above are checked
2. You can teach this phase to someone else
3. You understand not just *what* but *why*
4. You can read DSPy code comfortably

**Not ready yet?**
- Review ADRs for this phase
- Re-read DSPy reference code
- Redo exercises
- Discuss on Discord/forum

**Bonus Challenge**:
[Advanced challenge that requires deep understanding]
```

---

## 🛠️ Project Structure

```
mini-dspy/
├── README.md                           # Project overview
├── LEARNING_GUIDE.md                   # How to use this project
├── HISTORICAL_TIMELINE.md              # Maps commits to DSPy history
│
├── docs/
│   ├── adrs/                           # Architecture Decision Records
│   │   ├── 001-module-system.md
│   │   ├── 002-metaclass-pattern.md
│   │   ├── 003-parameter-tracking.md
│   │   ├── 004-callback-pattern.md
│   │   ├── 005-field-definitions.md
│   │   ├── 006-signature-basemodel.md
│   │   ├── 007-signature-metaclass.md
│   │   ├── 008-string-signatures.md
│   │   ├── 009-signature-manipulation.md
│   │   ├── 010-predict-forward.md
│   │   ├── 011-demo-management.md
│   │   ├── 012-state-serialization.md
│   │   ├── 013-chain-of-thought.md
│   │   ├── 014-adapter-pattern.md
│   │   ├── 015-chat-formatting.md
│   │   ├── 016-chat-parsing.md
│   │   ├── 017-callback-integration.md
│   │   ├── 018-base-lm-interface.md
│   │   ├── 019-litellm-integration.md
│   │   ├── 020-lm-caching.md
│   │   ├── 021-usage-tracking.md
│   │   ├── 022-settings-context.md
│   │   ├── 023-teleprompter-base.md
│   │   ├── 024-example-prediction.md
│   │   ├── 025-evaluate-framework.md
│   │   ├── 026-bootstrap-setup.md
│   │   ├── 027-bootstrap-execution.md
│   │   ├── 028-labeled-fewshot.md
│   │   ├── 029-public-api.md
│   │   ├── 030-retrieve-module.md
│   │   └── 031-end-to-end-rag.md
│   │
│   ├── comparisons/                    # Mini vs Real DSPy
│   │   ├── module-system-comparison.md
│   │   ├── signature-comparison.md
│   │   ├── predict-comparison.md
│   │   ├── adapter-comparison.md
│   │   ├── optimizer-comparison.md
│   │   └── performance-comparison.md
│   │
│   ├── diagrams/                       # Visual architecture
│   │   ├── execution-pipeline.svg
│   │   ├── module-composition.svg
│   │   ├── signature-creation.svg
│   │   ├── adapter-flow.svg
│   │   ├── optimization-flow.svg
│   │   └── object-model.svg
│   │
│   ├── checkpoints/                    # Learning checkpoints
│   │   ├── phase1-checkpoint.md
│   │   ├── phase2-checkpoint.md
│   │   ├── phase3-checkpoint.md
│   │   ├── phase4-checkpoint.md
│   │   ├── phase5-checkpoint.md
│   │   ├── phase6-checkpoint.md
│   │   └── phase7-checkpoint.md
│   │
│   └── references/                     # DSPy references
│       ├── papers.md                   # All related papers
│       ├── dspy-commits.md             # Key commits
│       ├── peps-referenced.md          # Python PEPs used
│       └── external-resources.md       # Other resources
│
├── src/
│   ├── __init__.py
│   │
│   ├── primitives/                     # Phase 1
│   │   ├── __init__.py
│   │   ├── base_module.py              # Commit 1.1
│   │   ├── module.py                   # Commit 1.2
│   │   ├── example.py                  # Phase 6
│   │   ├── prediction.py               # Phase 6
│   │   └── README.md                   # Phase 1 learning guide
│   │
│   ├── signatures/                     # Phase 2
│   │   ├── __init__.py
│   │   ├── field.py                    # Commit 2.1
│   │   ├── signature.py                # Commits 2.2, 2.3
│   │   ├── utils.py                    # Commits 2.4, 2.5
│   │   └── README.md
│   │
│   ├── predict/                        # Phase 3
│   │   ├── __init__.py
│   │   ├── parameter.py                # Marker class
│   │   ├── predict.py                  # Commits 3.1-3.3
│   │   ├── chain_of_thought.py         # Commit 3.4
│   │   └── README.md
│   │
│   ├── adapters/                       # Phase 4
│   │   ├── __init__.py
│   │   ├── base.py                     # Commit 4.1
│   │   ├── chat_adapter.py             # Commits 4.2, 4.3
│   │   └── README.md
│   │
│   ├── clients/                        # Phase 5
│   │   ├── __init__.py
│   │   ├── base_lm.py                  # Commit 5.1
│   │   ├── lm.py                       # Commits 5.2, 5.3
│   │   ├── cache.py                    # Part of 5.3
│   │   └── README.md
│   │
│   ├── teleprompt/                     # Phase 6
│   │   ├── __init__.py
│   │   ├── teleprompt.py               # Commit 6.1
│   │   ├── bootstrap.py                # Commits 6.4, 6.5
│   │   ├── vanilla.py                  # Commit 6.6
│   │   └── README.md
│   │
│   ├── evaluate/                       # Phase 6
│   │   ├── __init__.py
│   │   ├── evaluate.py                 # Commit 6.3
│   │   └── README.md
│   │
│   ├── retrievers/                     # Phase 7
│   │   ├── __init__.py
│   │   ├── retrieve.py                 # Commit 7.2
│   │   └── README.md
│   │
│   ├── utils/                          # Various phases
│   │   ├── __init__.py
│   │   ├── usage_tracker.py            # Commit 5.4
│   │   ├── settings.py                 # Commit 5.5
│   │   ├── callback.py                 # Commit 1.4
│   │   └── README.md
│   │
│   └── dsp/                            # Legacy compatibility
│       └── utils/
│           └── settings.py
│
├── tests/
│   ├── unit/                           # Unit tests per module
│   │   ├── test_base_module.py
│   │   ├── test_module.py
│   │   ├── test_signature.py
│   │   ├── test_predict.py
│   │   ├── test_adapter.py
│   │   ├── test_lm.py
│   │   ├── test_bootstrap.py
│   │   └── test_evaluate.py
│   │
│   ├── integration/                    # End-to-end tests
│   │   ├── test_simple_qa.py
│   │   ├── test_chain_of_thought.py
│   │   ├── test_optimization.py
│   │   └── test_rag_pipeline.py
│   │
│   ├── exercises/                      # Learning exercises
│   │   ├── phase1/
│   │   ├── phase2/
│   │   ├── phase3/
│   │   ├── phase4/
│   │   ├── phase5/
│   │   ├── phase6/
│   │   └── phase7/
│   │
│   └── solutions/                      # Exercise solutions
│       ├── phase1_exercise1.py
│       ├── phase1_exercise2.py
│       └── ...
│
├── benchmarks/                         # Performance tests
│   ├── phase1_benchmark.py
│   ├── phase2_benchmark.py
│   ├── phase3_benchmark.py
│   ├── phase4_benchmark.py
│   ├── phase5_benchmark.py
│   ├── phase6_benchmark.py
│   ├── phase7_benchmark.py
│   └── compare_with_dspy.py
│
├── examples/                           # Python programs to run
│   ├── 01_simple_qa.py
│   ├── 02_chain_of_thought.py
│   ├── 03_multi_step.py
│   ├── 04_optimization.py
│   ├── 05_rag_pipeline.py
│   └── 06_custom_module.py
│
├── tools/                              # Learning tools
│   ├── visualizer/
│   │   ├── show_module_tree.py
│   │   ├── show_signature.py
│   │   ├── show_execution.py
│   │   └── README.md
│   │
│   ├── debugger/
│   │   ├── step_through.py
│   │   ├── inspect_state.py
│   │   └── README.md
│   │
│   └── profiler/
│       ├── profile_execution.py
│       ├── compare_performance.py
│       └── README.md
│
├── requirements.txt                    # Dependencies
├── setup.py                            # Package setup
└── Makefile                            # Build commands
```

---

## 🎯 Commit Message Format

```
[Phase X.Y] Title - Historical Context

Brief description of what this commit implements.

DSPy Reference: path/to/file.py:100-200
Paper Reference: [Paper Name] Section X.Y
Historical Note: This mirrors DSPy v2.0's [feature] introduction

Design Decisions:
- Decision 1: [rationale]
  - Alternative considered: [why rejected]
- Decision 2: [rationale]
  - Trade-off: [what we sacrifice]

Simplifications from Real DSPy:
- Simplified [X] because [Y]
- Omitted [A] to focus on [B]

Learning Outcomes:
1. Understand [concept]
2. See how [pattern] enables [feature]
3. Compare [approach A] vs [approach B]

See docs/adrs/ADR-XXX.md for full decision record.
See docs/checkpoints/phaseX-checkpoint.md for exercises.

Tests: pytest tests/unit/test_[component].py
Benchmark: python benchmarks/phaseX_benchmark.py
```

---

## 📚 Key Learning Resources

### Papers (Chronological)

1. **[DSP: Demonstrate-Search-Predict](https://arxiv.org/abs/2212.14024)** (Dec 2022)
   - Original RAG framework
   - Bootstrap few-shot examples
   - Demonstrates-Search-Predict pipeline

2. **[DSPy: Compiling Declarative Language Model Calls](https://arxiv.org/abs/2310.03714)** (Oct 2023)
   - Declarative programming model
   - Signature abstraction
   - Teleprompter compilation

3. **[DSPy Assertions](https://arxiv.org/abs/2312.13382)** (Dec 2023)
   - Computational constraints
   - Self-refining pipelines
   - Assertion-driven optimization

4. **[MIPRO: Multi-stage Instruction and Demonstration Optimization](https://arxiv.org/abs/2406.11695)** (Jun 2024)
   - Instruction optimization
   - Demo selection strategies
   - Multi-objective optimization

5. **[GEPA: Reflective Prompt Evolution](https://arxiv.org/abs/2507.19457)** (Jul 2025)
   - Reflective optimization
   - Outperforms RL
   - Latest optimizer

### DSPy Documentation

- **Official Docs**: https://dspy.ai
- **GitHub**: https://github.com/stanfordnlp/dspy
- **Discord**: https://discord.gg/XCGy2WDCQB

### Related Frameworks

- **PyTorch**: Module system inspiration
- **Pydantic**: Type validation
- **LiteLLM**: Multi-provider LM support
- **scikit-learn**: Fit/transform pattern

---

## 🔧 Development Tools

### 1. Module Tree Visualizer
```bash
python tools/visualizer/show_module_tree.py examples/rag_pipeline.py
```

**Output**:
```
RAG(Module)
├── generate_query: ChainOfThought(Predict)
│   └── signature: GenerateSearchQuery
│       ├── inputs: ['question']
│       └── outputs: ['query', 'rationale']
├── retrieve: Retrieve
│   └── k: 3
└── generate_answer: ChainOfThought(Predict)
    └── signature: GenerateAnswer
        ├── inputs: ['context', 'question']
        └── outputs: ['answer', 'rationale']
```

### 2. Execution Tracer
```bash
python tools/debugger/step_through.py examples/simple_qa.py
```

**Interactive debugging**:
- Step through execution
- Inspect module state
- See LM calls
- View prompts/responses

### 3. Performance Profiler
```bash
python tools/profiler/compare_performance.py examples/rag_pipeline.py
```

**Output**:
```
Component         | Mini-DSPy | Real DSPy | Ratio
------------------|-----------|-----------|-------
Module creation   | 0.1ms     | 0.5ms     | 5.0x
Signature parse   | 1.2ms     | 0.8ms     | 0.67x
LM call           | 250ms     | 240ms     | 0.96x
Adapter format    | 2.5ms     | 1.8ms     | 0.72x
Total             | 254ms     | 243ms     | 0.96x
```

---

## ✅ What's New in This Approach?

| Enhancement | Benefit |
|-------------|---------|
| **Architecture Decision Records** | Understand *why*, not just *what* |
| **Historical Timeline** | See DSPy's evolution from DSP to v3 |
| **Learning Checkpoints** | Active learning with exercises |
| **Comparative Analysis** | Explicit mini vs real DSPy differences |
| **Paper References** | Connect code to research |
| **Interactive Tools** | Visualize execution, inspect state |
| **Performance Benchmarks** | Understand optimization trade-offs |
| **Incremental Commits** | Each commit is a learning milestone |

---

## 🚀 Getting Started

### Prerequisites
```bash
# Python 3.10+
python --version

# Install dependencies
pip install pydantic litellm pytest
```

### Start Learning

1. **Read this framework** (you're doing it!)
2. **Set up project structure**:
   ```bash
   mkdir mini-dspy && cd mini-dspy
   # Follow structure above
   ```
3. **Start with Phase 1, Commit 1.1**:
   - Read `docs/adrs/001-module-system.md`
   - Implement `src/primitives/base_module.py`
   - Write tests
   - Complete checkpoint

4. **Iterate through phases**

---

## 🎓 Learning Philosophy

### The "Why" Matters Most

This framework emphasizes:
- **Understanding over implementation**: Know why designs exist
- **Historical context**: See how ideas evolved
- **Trade-off analysis**: Every decision has costs
- **Comparative learning**: Mini vs Real teaches judgment
- **Active practice**: Build, don't just read

### Success Criteria

You've succeeded when you can:
1. **Explain** design decisions without notes
2. **Critique** DSPy's architecture thoughtfully
3. **Extend** mini-DSPy with new features
4. **Read** real DSPy code comfortably
5. **Teach** DSPy concepts to others

---

## 📊 Progress Tracking

| Phase | Commits | Status | Checkpoint Complete |
|-------|---------|--------|---------------------|
| Phase 1: Module System | 4 | ⬜ Not Started | ⬜ |
| Phase 2: Signatures | 5 | ⬜ Not Started | ⬜ |
| Phase 3: Predict | 4 | ⬜ Not Started | ⬜ |
| Phase 4: Adapters | 4 | ⬜ Not Started | ⬜ |
| Phase 5: LM Clients | 5 | ⬜ Not Started | ⬜ |
| Phase 6: Optimizers | 6 | ⬜ Not Started | ⬜ |
| Phase 7: Integration | 3 | ⬜ Not Started | ⬜ |

**Total**: 31 commits, 7 phases, 3,000-5,000 lines of Python

---

## 🎯 Final Goal

Build a working DSPy implementation that can:

```python
# Define a signature
class QA(dspy.Signature):
    """Answer questions using context."""
    context: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

# Create a program
class RAG(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=3)
        self.answer = dspy.ChainOfThought(QA)

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.answer(context=context, question=question)

# Configure LM
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# Use it
rag = RAG()
result = rag(question="What is DSPy?")
print(result.answer)

# Optimize it
optimizer = dspy.BootstrapFewShot(metric=my_metric)
optimized_rag = optimizer.compile(rag, trainset=examples)

# Evaluate
evaluator = dspy.Evaluate(devset=test_data)
score = evaluator(optimized_rag)
```

---

## 🚀 Ready to Begin?

**Next Step**: Create the project structure and implement Phase 1, Commit 1.1!

See you in `docs/adrs/001-module-system.md`! 🎓
