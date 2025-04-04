# Combined Model Evaluation Report

## Part 1: Custom Influence-Based Evaluation

### Model Comparison Summary

| Metric       |   Base Model |   Fine-tuned Model |   Improvement |   Relative Improvement (%) |
|:-------------|-------------:|-------------------:|--------------:|---------------------------:|
| BERT_SCORE   |     0.823339 |            0.87108 |     0.0477415 |                    5.79853 |
| SEMANTIC_SIM |     0.381179 |            0.57943 |     0.19825   |                   52.0097  |


> **Metric Explanations:**
> - **BERT_SCORE**: Measures semantic similarity using contextual embeddings. Higher values indicate better alignment of meaning between model outputs and reference texts.
> - **SEMANTIC_SIM**: Measures cosine similarity between sentence embeddings of model outputs and reference texts, capturing overall semantic similarity.
> Both metrics range from 0 to 1, with higher values indicating better performance.

## Part 2: Standardized Benchmarks

### Benchmark Results for: base_tinyllama

| Benchmark | Overall Score |
|-----------|---------------|
| MMLU | 0.2473 |

### Benchmark Results for: finetuned_tinyllama

| Benchmark | Overall Score |
|-----------|---------------|
| MMLU | 0.2410 |

#### MMLU Task-Specific Scores

*MMLU task scores table found but columns mismatch. Displaying raw table:*

| Task              |    Score |
|:------------------|---------:|
| professional_law  | 0.237288 |
| business_ethics   | 0.3      |
| international_law | 0.239669 |


## Part 3: Training Data Influence Analysis

For detailed analysis, see: [influential_examples.md](../comparison/influential_examples.md)

### Sample of Influence Analysis

```markdown
# Influential Training Examples Analysis

## Prompt 1: "What are some examples of isocyanate-reactive compounds used in the production of tin-free polyurethane foam?"

**Generated completion:** Examples of iso
High density fiberboard consists essentially of fiber with a diameter ranging from 0.1 to 4 mm, which are bonded together by means of isophorone diisocyanates and optionally further crosslinking agents.

### Most Influential Training Examples

| Rank | Score | Example Text (Truncated to 500 chars) |
|------|-------|---------------------------------------|
| 1 | 3244032.000000 |  What are some examples of isocyanate-reactive compounds used in the production of tin-free polyurethane foam? |
| 2 | 1662976.000000 |  What are the advantages of using alkali metal carboxylates as catalysts in the production of tin-free polyurethane foam? |
| 3 | 1589248.000000 |  What is the role of alkali metal carboxylates in the production of tin-free polyurethane foam? |
| 4 | 1236992.000000 |  What is the catalyst used in the production of the tin-free polyurethane foam? |
| 5 | 1187840.000000 |  What are some specific uses of tin-free polyurethane foam? |
| 6 | 1171456.000000 | What are some examples of isocyanate-reactive compounds in the invention? |
| 7 | 1097728.000000 | What are the components of the low temperature curing coating composition? |
| 8 | 1089536.000000 | What are some of the applications of isophorone diamine (IPDA)? |
| 9 | 1032192.000000 | What are the characteristics of the open-cell rigid foam? |
| 10 | 991232.000000 |  What is the density range of the tin-free polyurethane foam? |

---

## Prompt 2: "What is the composition of the acrylic copolymer used in the multilayer composite and underlayment for roof assembly?"

**Generated completion:** The acrylic copolysurface is a combination of at least one (meth)acrylate component, which can be either an emulsion or a solution, along with a polyisocyanate component.

### Most Influential Training Examples

...
```


## Conclusion

This report combines custom influence-based analysis with standardized benchmarks (e.g., DeepEval). The combined evaluation provides a view of the model's performance and the impact of fine-tuning.

Key findings:

- Custom Comparison: The fine-tuned model showed the greatest improvement in **SEMANTIC_SIM** with an absolute improvement of 0.1983 (52.01%).
- Benchmarks: The fine-tuned model achieved an overall score of **0.2410** on the MMLU benchmark.
- The influence analysis identifies training examples with significant impact on specific prompt outputs.
