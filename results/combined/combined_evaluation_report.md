# Combined Model Evaluation Report

## Part 1: Custom Influence-Based Evaluation

### Model Comparison Summary

| Metric       |   Base Model |   Fine-tuned Model |   Improvement |   Relative Improvement (%) |
|:-------------|-------------:|-------------------:|--------------:|---------------------------:|
| BERT_SCORE   |     0.834319 |           0.114063 |     -0.720256 |                   -86.3286 |
| SEMANTIC_SIM |     0.390285 |           0.102898 |     -0.287386 |                   -73.6351 |


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
| MMLU | 0.2467 |

#### MMLU Task-Specific Scores

*MMLU task scores table found but columns mismatch. Displaying raw table:*

| Task              |    Score |
|:------------------|---------:|
| professional_law  | 0.243807 |
| business_ethics   | 0.3      |
| international_law | 0.239669 |


## Part 3: Training Data Influence Analysis

For detailed analysis, see: [influential_examples.md](../comparison/influential_examples.md)

### Sample of Influence Analysis

```markdown
# Influential Training Examples Analysis

## Prompt 1: "What are some examples of isocyanate-reactive compounds used in the production of tin-free polyurethane foam?"

**Generated completion:** 

### Most Influential Training Examples

| Rank | Score | Example Text (Truncated to 500 chars) |
|------|-------|---------------------------------------|
| 1 | 3424256.000000 |  What are some examples of isocyanate-reactive compounds used in the production of tin-free polyurethane foam? |
| 2 | 1777664.000000 |  What are the advantages of using alkali metal carboxylates as catalysts in the production of tin-free polyurethane foam? |
| 3 | 1662976.000000 |  What is the role of alkali metal carboxylates in the production of tin-free polyurethane foam? |
| 4 | 1441792.000000 |  What are some specific uses of tin-free polyurethane foam? |
| 5 | 1138688.000000 |  What is the density range of the tin-free polyurethane foam? |
| 6 | 1130496.000000 |  What is the catalyst used in the production of the tin-free polyurethane foam? |
| 7 | 983040.000000 | What are some examples of isocyanate-reactive compounds in the invention? |
| 8 | 917504.000000 |  How is the tin-free polyurethane foam obtained in the manufacturing process? |
| 9 | 585728.000000 |  What is the composition of the molybdenum and tungsten-free hardenable mixture? |
| 10 | 565248.000000 |  What is the main difference between the tin-containing polyurethane foam and the tin-free polyurethane foam? |

---

## Prompt 2: "What is the composition of the acrylic copolymer used in the multilayer composite and underlayment for roof assembly?"

**Generated completion:** What is the ratio of polyisocyanate to other components present in the aclylic copolyomer used in a multilayerdo-layer composite and/or as an underlayement fo rroofassembly.

### Most Influential Training Examples

| Rank | Score | Example Text (Truncated to 500 chars) |
...
```


## Conclusion

This report combines custom influence-based analysis with standardized benchmarks (e.g., DeepEval). The combined evaluation provides a view of the model's performance and the impact of fine-tuning.

Key findings:

- Custom Comparison: The fine-tuned model showed the greatest improvement in **SEMANTIC_SIM** with an absolute improvement of -0.2874 (-73.64%).
- Benchmarks: The fine-tuned model achieved an overall score of **0.2467** on the MMLU benchmark.
- The influence analysis identifies training examples with significant impact on specific prompt outputs.
