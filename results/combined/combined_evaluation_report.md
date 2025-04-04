# Combined Model Evaluation Report

## Part 1: Custom Influence-Based Evaluation

### Model Comparison Summary

| Metric   |   Base Model |   Fine-tuned Model |   Improvement |   Relative Improvement (%) |
|:---------|-------------:|-------------------:|--------------:|---------------------------:|
| BLEU1    |    0.0881698 |         0.0342466  |   -0.0539232  |                   -61.1584 |
| BLEU2    |    0.0143247 |         0.0109047  |   -0.00342004 |                   -23.8751 |
| ROUGE1   |    0.212696  |         0.0785621  |   -0.134134   |                   -63.0636 |
| ROUGE2   |    0.0321645 |         0.00961538 |   -0.0225492  |                   -70.1056 |
| ROUGEL   |    0.119244  |         0.0455433  |   -0.0737009  |                   -61.8067 |

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

## Prompt 1: "Explain quantum computing"

**Generated completion:** as a computational technique that relies on the principles of classical physics, but with increased levels of parallelism and specialized hardware.

### Most Influential Training Examples

| Rank | Score | Example Text (Truncated to 500 chars) |
|------|-------|---------------------------------------|
| 1 | 3719168.000000 |  Explain the components of the polyurethane-forming system in the invention. |
| 2 | 236544.000000 | How is a robotic arm defined? |
| 3 | 216064.000000 |  What are the components used to produce new polyurethane shoe soles? |
| 4 | 198656.000000 | What are the monomer compounds used to obtain the new polymer? |
| 5 | 195584.000000 | What is the purpose of learning in a neural network? |
| 6 | 192512.000000 | What are the components of the harmful animal expelling plastic composition? |
| 7 | 188416.000000 | How does the system overcome lack of cure in thin film blend areas? |
| 8 | 187392.000000 | What is the capability of a neural network? |
| 9 | 182272.000000 | What does the computer-implemented method involve for controlling sorting of plastic compounds? |
| 10 | 180224.000000 | What are some examples of polymeric plasticizers that can be used in the asphalt composition? |

---

## Prompt 2: "How does machine learning work?"

**Generated completion:** 

### Most Influential Training Examples

| Rank | Score | Example Text (Truncated to 500 chars) |
...
```


## Conclusion

This report combines custom influence-based analysis with standardized benchmarks (e.g., DeepEval). The combined evaluation provides a view of the model's performance and the impact of fine-tuning.

Key findings:

- Custom Comparison: The fine-tuned model showed the greatest improvement in **BLEU2** with an absolute improvement of -0.0034 (-23.88%).
- Benchmarks: The fine-tuned model achieved an overall score of **0.2467** on the MMLU benchmark.
- The influence analysis identifies training examples with significant impact on specific prompt outputs.
