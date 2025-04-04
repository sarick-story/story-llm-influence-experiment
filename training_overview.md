```mermaid
flowchart TD
    %% Subgraph: Huggingface
    subgraph "Huggingface"
        base_model["TinyLlama 1.1B Model"]:::sourceStyle
        dataset["Patent QA Dataset"]:::sourceStyle
    end

    %% Subgraph: Training Module
    subgraph "Training Module"
        training["Training Module"]:::trainingStyle
    end

    %% Subgraph: Influence Analysis
    subgraph "Influence Analysis"
        factors["Influence Analysis - Factors"]:::analysisStyle
        scores["Influence Analysis - Scores"]:::analysisStyle
    end

    %% Subgraph: Evaluation
    subgraph "Evaluation"
        attribution["Attribution Analysis"]:::evaluationStyle
        deepeval_mmlu["DeepEval - MMLU (LAW, ETHICS, INT'L LAW)"]:::evaluationStyle
        metrics["BERT-Score/Semantic Similarity Scores Scores"]:::evaluationStyle
    end

    %% External Dependencies
    subgraph "External Dependencies"
        externalKron["Kronfluence"]:::externalStyle
        externalDeep["DeepEval Library"]:::externalStyle
        wandb_service["WandB Service"]:::externalStyle
    end

    %% Data Sources to Training
    base_model --> training
    dataset -- Data Cleaning & Preparation --> training

    %% Core Connections
    training --> factors
    training --> scores
    factors --> scores

    %% Connections to Evaluation
    factors --> attribution
    scores --> attribution
    training --> deepeval_mmlu
    training --> metrics

    %% External Dependencies Connections
    factors --- externalKron
    scores --- externalKron
    deepeval_mmlu --- externalDeep

    %% WandB Connections (dotted for logging/monitoring)
    training -.-> wandb_service
    factors  -.-> wandb_service
    scores   -.-> wandb_service
    metrics -.-> wandb_service
    deepeval_mmlu -.-> wandb_service

    %% Styles
    classDef configStyle fill:#FFEB3B,stroke:#F57F17,stroke-width:2px;
    classDef orchestratorStyle fill:#8BC34A,stroke:#33691E,stroke-width:2px;
    classDef trainingStyle fill:#64B5F6,stroke:#0D47A1,stroke-width:2px;
    classDef analysisStyle fill:#E1BEE7,stroke:#6A1B9A,stroke-width:2px;
    classDef evaluationStyle fill:#FFCDD2,stroke:#B71C1C,stroke-width:2px;
    classDef utilityStyle fill:#CFD8DC,stroke:#37474F,stroke-width:2px;
    classDef externalStyle fill:#D7CCC8,stroke:#4E342E,stroke-width:2px;
    classDef sourceStyle fill:#A5D6A7,stroke:#2E7D32,stroke-width:2px;
```

# Training Process Overview

## Fine-tuning for Training Data Attribution

We use fine-tuning for training data attribution because it allows us to directly measure how specific training examples influence the model's predictions. By fine-tuning a pre-trained model on our dataset, we can track the gradient information and influence factors that connect individual training examples to the model's behavior. This approach enables us to attribute model outputs to specific training inputs, providing transparency into which examples had the most significant impact on particular predictions.

## Data Preparation and Cleaning

For our patent classification dataset, data preparation includes:

1. **Text normalization**: Standardizing text format, case, and special characters
2. **Tokenization**: Converting raw text into tokens the model can process
3. **Length adjustment**: Truncating or padding sequences
4. **Format conversion**: Structuring data into question-answer format for the model

[Dataset](https://huggingface.co/datasets/XiaoluBELLA/Patent_classification_QA)

[Model](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T)

## Key Hyperparameters and Their Significance

When fine-tuning a language model for data attribution, several hyperparameters significantly affect training efficiency and the quality of attribution analysis:

- **Learning rate**: Controls how quickly the model adapts to new data. Too high may cause instability, too low may result in slow convergence.
- **Batch size**: Affects memory usage and training speed. Larger batches provide more stable gradients but require more memory.
- **Number of epochs**: Determines how many times the model sees the entire dataset.
- **Weight decay**: Regularization technique to prevent overfitting by penalizing large weights.
- **Gradient accumulation**: Allows for effective larger batch sizes by accumulating gradients over multiple forward passes.

## Training Metrics

During training, we primarily monitor **loss values**, which indicate how well the model is learning from the data. A decreasing loss suggests the model is improving its predictions. For language models, we typically use cross-entropy loss, which measures the difference between predicted token probabilities and actual tokens.

We also track:
- **Learning rate changes** (especially with schedulers like cosine with restarts)
- **GPU memory usage** to ensure efficient resource utilization
- **Training speed** (samples processed per second)

## Training Time and Resources

Training our TinyLlama model took approximately 20 minutes on an NVIDIA A100 GPU. This relatively short training time is achieved through careful optimization of training parameters and taking advantage of the GPU's capabilities.

## Training Configuration Parameters

Our `config.yaml` includes several parameters specifically optimized for training speed and GPU utilization:

```yaml
models:
  base:
    name: "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    id: "base_tinyllama"
  finetuned:
    path: "./tinyllama_1b_model"
    id: "finetuned_tinyllama"

general:
  max_length: 512
  use_flash_attention: false

dataset:
  num_samples: 16000
  format: "qa"
  input_column: "Input"
  output_column: "Output"
```

The TinyLlama model (1.1B parameters) was selected as it provides a good balance between capability and training speed. We set a reasonable max sequence length of 512 tokens to limit memory usage while still accommodating most patent descriptions.

## Training Arguments Explained

```python
training_args = TrainingArguments(
    output_dir=model_output_path,  # Directory where model checkpoints are saved
    num_train_epochs=1,  # Single epoch training is sufficient for our attribution analysis
    per_device_train_batch_size=8,  # 8 samples processed at once per GPU
    save_steps=100,  # Save a checkpoint every 100 steps
    logging_steps=20,  # Log metrics every 20 steps for monitoring
    learning_rate=2e-5,  # Conservative learning rate to avoid catastrophic forgetting
    weight_decay=0.01,  # Regularization to prevent overfitting
    logging_dir="./logs",  # Directory for storing logs
    lr_scheduler_type="cosine_with_restarts",  # Learning rate schedule that reduces LR over time with periodic resets
    gradient_accumulation_steps=1,  # No gradient accumulation, process each batch directly
    max_grad_norm=None if use_bf16 else 1.0,  # Disabled gradient clipping when using BF16 precision
    bf16=use_bf16,  # Use bfloat16 precision when available for faster training
    gradient_checkpointing=False,  # Disabled to improve speed since we have sufficient memory
    optim="adamw_torch_fused",  # Optimized implementation of AdamW for faster computation
    dataloader_num_workers=4,  # Parallel data loading with 4 workers
    ddp_find_unused_parameters=False,  # Optimization for distributed training
    dataloader_drop_last=True,  # Drop incomplete batches for consistent batch sizes
    remove_unused_columns=True,  # Memory optimization by removing unneeded columns
    report_to="wandb" if wandb.run is not None else None,  # Report metrics to Weights & Biases if configured
)
```

### Training Run Visualization

The training process was monitored using Weights & Biases (WandB), allowing us to track key metrics in real-time. You can view the detailed training run at [Weights & Biases Dashboard](https://wandb.ai/sarick-shah-pip-labs/influence-llm/runs/i83j2b3u?nw=nwusersarickshah).

Below are visualizations from the training process:

![Training Metrics Visualization](train.png)

The visualization above shows the progression of loss and learning rate throughout the training process, demonstrating how the model gradually improved as training progressed.

![Additional Training Metrics](train2.png)

This second visualization provides additional insights into the model's training dynamics, showing metrics such as gradient norms and optimization statistics that help verify the stability of the training process.

## Influence Factor Computation using Kronfluence

Once the model is fine-tuned, we need to understand how individual training examples influence its behavior. Ideally, we'd measure influence by seeing how much a test prediction changes if we remove a specific training example and retrain. This is computationally infeasible.

Influence functions provide a mathematical shortcut. Conceptually, the influence of a training example (`z_m`) on a test example (`z_c`) can be estimated as:

```
Influence ≈ (How test loss changes with model weights) * (Inverse of model sensitivity) * (How training loss changes with model weights)
```

Mathematically, this involves the gradients of the loss functions and the inverse of the Hessian matrix ($\mathcal{H}^{-1}$):

\[ \mathcal{I}_{\theta} (z_c, z_m) = - \nabla_{\theta} \mathcal{L}(z_c, \theta) \ \mathcal{H}^{-1} \ \nabla_{\theta} \mathcal{L}(z_m, \theta) \]

The **Hessian** ($\mathcal{H}$) measures the curvature or sensitivity of the model's training loss landscape. It tells us how the gradient itself changes as model weights change. The challenge is that for large models like TinyLlama (1.1B parameters), the Hessian is enormous (billions x billions matrix), making it impossible to compute and invert directly.

### EKFAC: Approximating the Inverse Hessian

We use the [Kronfluence library](https://github.com/pomonam/kronfluence), which employs the **Eigenvalue-corrected Kronecker-Factored Approximate Curvature (EKFAC)** method to approximate the crucial $\mathcal{H}^{-1}$ term efficiently.

1.  **Kronecker-Factored Approximation (K-FAC)**: This first step simplifies the problem by assuming gradients are independent across different layers and making further approximations within layers. This allows approximating the Hessian (specifically, the Fisher Information Matrix, which is equivalent for our loss function) with smaller, block-diagonal matrices based on Kronecker products. These smaller blocks are much easier to handle computationally.

2.  **Eigenvalue Correction (E)**: K-FAC's approximations can introduce errors. The "E" in EKFAC corrects for this by computing the eigenvalues (which represent the magnitude of curvature in different directions) of the K-FAC blocks and adjusting them to better match the eigenvalues of the true, full Hessian. This significantly improves the approximation's accuracy without needing the full Hessian.

### The `fit_all_factors` Process

The `modules/analysis/factors/compute.py` script orchestrates this using Kronfluence's `Analyzer`.

-   **Task Definition (`LanguageModelingTask`)**: We define a `LanguageModelingTask` (from `modules/analysis/factors/task.py`). This tells Kronfluence how to calculate the loss for our specific model and which parts of the model (modules) are relevant for the influence analysis. Based on our `config.yaml`, we configure it to analyze the MLP components (`gate_proj`, `up_proj`, `down_proj`) of *all* transformer layers.

-   **Factor Computation**: The `analyzer.fit_all_factors` function iterates through a subset of the training data (`analysis_samples` = 16000 from `config.yaml`). For each batch, it calculates the EKFAC factors (the approximated $\mathcal{H}^{-1}$ information) for the specified MLP modules across all layers. Performance is optimized using settings in `config.yaml` (`factors.batch_size`, `factors.num_workers`, `factors.performance_options`) like parallel data loading, data partitioning, and mixed-precision computation (`bfloat16`).

**Output**: The process saves the computed EKFAC factors (e.g., `tinyllama_1b_factors_all_layers.pt`) in the `results/influence/factors/` directory. These factors represent the pre-computed approximation of the model's sensitivity (inverse Hessian) and are essential for the next step: calculating specific influence scores efficiently.

## Inspecting Influence Factors

After computing the EKFAC factors, it's useful to inspect them to understand their structure and what they tell us about the model's sensitivity. The `modules/analysis/factors/inspect.py` script handles this visualization.

**Purpose**: The main goal is to visualize the computed approximation of the model's sensitivity (specifically, the EKFAC representation of the inverse Hessian) for a particular layer. This helps us understand how well the EKFAC method captured the important directions of curvature in the loss landscape and reveals properties about the model's parameter space relevant to the task.

**Why Inspect a Specific Layer?**
We often focus on inspecting factors from a specific layer, typically one of the later layers (like layer 21, the default `inspection_layer` in our `config.yaml`). Deeper layers in a transformer often learn more abstract and task-specific representations, so their sensitivity patterns can be particularly informative about how the model handles the task (in our case, patent classification QA).

**How it Works**: 
The script loads the pre-computed EKFAC factors (specifically the 'lambda matrix', which relates to the curvature) for the specified layer. It then calculates the *eigenvalues* of this matrix. Eigenvalues represent the magnitude of sensitivity along different directions in the parameter space. Finally, it generates plots:

1.  A heatmap of the lambda matrix itself.
2.  A plot showing the distribution of eigenvalues (sorted from largest to smallest).

**Interpreting the Results**:

*   **Lambda Matrix Heatmap**: Shows the structure of the sensitivity approximation (related to the EKFAC factors) for a specific component (likely layer 21's MLP `down_proj` layer). The axes represent input and output feature dimensions for that layer. The color indicates the sensitivity value for interactions between specific input/output features.

    *   **Interpretation**: The heatmap shows that most values are near zero (light grey), suggesting low sensitivity for many feature interactions within this layer. However, the stronger colors (red/blue) concentrated along the right and bottom edges indicate that the model's sensitivity is highly dependent on specific interactions involving higher-indexed input and output features of this layer. This structured, non-uniform pattern suggests the EKFAC approximation captures specific sensitivities rather than just noise, highlighting the feature interactions most critical to the model's curvature at this point.

    ![Lambda Matrix for Layer 21](results/influence/layer_21/lambda_matrix.png)

*   **Sorted Lambda Values Plot**: Another way to visualize the factor structure is to plot all the values from the lambda matrix for a specific module, sorted from smallest to largest. The plot below for layer 21's `down_proj` module uses a symmetric log scale (`symlog`) on the y-axis to handle positive and negative values near zero.

    *   **Interpretation**: This plot clearly shows that a vast majority of the values in the lambda matrix for this layer component are extremely close to zero (near 10^-5 or smaller). Only a very small fraction of the values have significantly larger magnitudes, indicated by the sharp upward curve on the far right. This confirms the "sparse" nature of the sensitivity captured by the EKFAC factors for this layer.

    *   This chart effectively displays the sensitivity or curvature calculated by the EKFAC method for this specific part of the model (`model.layers.21.mlp.down_proj`). Conceptually, this sensitivity relates to different "directions" within the model's vast space of possible settings (parameters). The horizontal axis represents these different sensitivity components sorted by magnitude, while the vertical axis (on a log scale) shows the sensitivity measure. The plot shows that for most components (the long flat part at the bottom), the model is not very sensitive; changes related to these components don't drastically affect its output curvature. However, there are a few components (the sharp upward curve on the right) where the model is extremely sensitive; changes related to these specific components significantly impact performance curvature. This information, especially the spread from very small to very large values, is crucial for the influence function calculations performed by Kronfluence, as it helps weigh how much impact individual training examples had on the final model by relating training data changes to these sensitive components.

    ![Sorted Lambda Values for Layer 21 down_proj](results/influence/layer_21/sorted_lambda_values_model.layers.21.mlp.down_proj.png)

## Influence Score Computation

Having computed and inspected the EKFAC factors (our approximation of model sensitivity, $\mathcal{H}^{-1}$), the next step is to calculate the actual influence scores. These scores quantify the impact of individual training data points on the model's behavior for specific queries or prompts we are interested in. The `modules/analysis/scores/compute.py` script manages this process.

**The Goal**: We want to estimate, for a given query (e.g., a specific prompt from `prompts.json`), which training examples were most influential (positively or negatively) on the model's output for that query.

**How it Works (Conceptually)**:
Recall the influence function formula:

\[ \mathcal{I}_{\theta} (z_c, z_m) \approx - \underbrace{\nabla_{\theta} \mathcal{L}(z_c, \theta)}_{\text{Query Gradient}} \ \underbrace{\mathcal{H}^{-1}}_{\text{Factors}} \ \underbrace{\nabla_{\theta} \mathcal{L}(z_m, \theta)}_{\text{Train Gradient}} \]

1.  **Load Factors**: The script first loads the pre-computed EKFAC factors (the $\mathcal{H}^{-1}$ approximation saved in the previous step, e.g., `tinyllama_1b_factors_all_layers.pt`).
2.  **Prepare Data**: It loads:
    *   The **training dataset** (the same `analysis_samples` subset used for factor computation).
    *   The **query dataset** (created from `prompts.json` or potentially from model-generated answers if specified).
3.  **Compute Gradients**: Using the fine-tuned model and the `LanguageModelingTask` (which defines how loss is calculated), Kronfluence computes:
    *   The gradient of the loss for each **query example** ($\\nabla_{\theta} \mathcal{L}(z_c, \theta)$).
    *   The gradient of the loss for each **training example** ($\\nabla_{\theta} \mathcal{L}(z_m, \theta)$).
4.  **Combine Components**: Kronfluence efficiently combines the query gradients, the loaded EKFAC factors, and the training gradients according to the formula to calculate the influence score for *every pair* of query and training examples.

**Key Configuration (`config.yaml` Scores Section)**:

*   `scores_name`: Name for the saved scores file (e.g., `tinyllama_prompt_scores`).
*   `query_gradient_rank`: This allows using a low-rank approximation for the query gradients. A lower rank speeds up computation significantly, especially with many queries, at the cost of some precision. We use a rank of 64.
*   `train_batch_size`: How many training example gradients are processed together (set to 4 for memory efficiency).

**Output**: The result is a large tensor (matrix) of influence scores, saved in the scores output directory (e.g., `results/influence/scores/tinyllama_prompt_scores.npz`). Each element `(i, j)` in this matrix represents the estimated influence of training example `j` on the model's loss for query example `i`. These scores are the key to understanding data attribution.

## Evaluating Model Performance and Influence

After fine-tuning and analyzing influence factors/scores, the final step is to evaluate the model's performance and understand how the training data influences specific outputs. Our pipeline uses two main evaluation approaches orchestrated by `main.py` when running the `evaluate --type all` command.

### Model Completion Comparison for Sample Prompts

Below is a comparison of completions between the base TinyLlama model and our fine-tuned model for several patent-related queries:

| Prompt | Expected Completion | Base Model Completion | Fine-tuned Model Completion |
|--------|---------------------|----------------------|----------------------------|
| What are some examples of isocyanate-reactive compounds used in the production of tin-free polyurethane foam? | Isocyanate-reactive compounds used in the production of tin-free polyurethane foam include polyisocyanates and compounds containing isocyanate-reactive hydrogen atoms. | The term "isocyanates" refers to a class of organic molecules that contains two nitrogen atoms bonded together. They're commonly found in plastics, adhesives and sealants, but they can also be made into industrial chemicals. Some common examples include acrylate polymers (aka acrylic acid), glycidyl ethers (acid esters) like MDI or diethylenetriamine penta | Examples of iso\nHigh density fiberboard consists essentially of fiber with a diameter ranging from 0.1 to 4 mm, which are bonded together by means of isophorone diisocyanates and optionally further crosslinking agents. |
| What is the composition of the acrylic copolymer used in the multilayer composite and underlayment for roof assembly? | The composition of the acrylic copolymer includes monomer A, monomer B, and monomer C, which are polymerized in an amount from 90-99.5 wt.%, along with a crosslinker in an amount of 0.5-10 wt.%. | 2. Which was first discovered in 1940 by Herman P Kroger (1895 – 1 September 1… Read More › | The acrylic copolysurface is a combination of at least one (meth)acrylate component, which can be either an emulsion or a solution, along with a polyisocyanate component. |
| What is the density range of the flame-protected polyurethane foam? | The flame-protected polyurethane foam has a density range of 5 to 50 g/L. | A: The density of flameproof polyurea will vary depending on factors such as product grade, manufacturing process and weather. In general, the higher a grade number or an OEK 8170 certified material, the more resistant it is to fire, but there may be some variation in its thickness due to this factor. It's always best to check the documentation supplied with your purchased item for specific details about how dense it is before specifying its use | The flameprotective polyurethanefoam has a density ranging from 20 to 145 kg/m3. |

### 1. Custom Evaluation and Influence Analysis

This part focuses on comparing the fine-tuned model against the base model using our specific prompts and analyzing the influence scores related to these prompts.

*   **Answer Generation**: The `modules.evaluation.custom.generate_model_answers` script takes the prompts defined in `prompts.json`, feeds them to both the base (`TinyLlama/...`) and the fine-tuned (`./tinyllama_1b_model`) models, and saves their generated answers (likely to `results/generated/`).

### 2. Standardized Benchmarking with DeepEval and MMLU

To gauge the model's capabilities on a broader, standardized set of tasks, we use the [DeepEval framework](https://docs.confident-ai.com/).

*   **DeepEval**: An open-source library designed for evaluating Large Language Models (LLMs) across various metrics and benchmarks. It provides tools to assess aspects like reasoning, knowledge, and safety.
*   **MMLU (Massive Multitask Language Understanding)**: A popular benchmark included within DeepEval. MMLU measures a model's knowledge across a diverse range of subjects, typically using multiple-choice questions. It aims to assess a model's ability to recall facts and reason across different domains.
*   **Our Configuration (`config.yaml`)**: We configure `modules.evaluation.deepeval.deepeval_runner.py` to run the MMLU benchmark specifically on the fine-tuned model (`eval_finetuned_model: true`, `eval_base_model: false`). We selected specific MMLU tasks relevant to our patent/legal domain:
    *   `PROFESSIONAL_LAW`
    *   `BUSINESS_ETHICS`
    *   `INTERNATIONAL_LAW`
    These tasks test the model's understanding in areas pertinent to patent classification and related legal/business concepts. The results are saved in `results/deepeval/`.

#### How the MMLU Benchmark Works

The MMLU benchmark evaluates models using a standardized methodology:

1. **Multiple-Choice Format**: Each question is presented with four possible answers (A, B, C, or D). The model must select the correct option, making this a classification task rather than an open-ended generation task.

2. **Few-Shot Learning**: The benchmark uses a few-shot evaluation approach, where the model is given several example questions and answers (typically 5 examples) before being asked to answer the test question. This tests the model's ability to recognize patterns and adapt to the task format without explicit fine-tuning.

3. **Diverse Subject Areas**: The full MMLU benchmark contains 57 subjects across STEM, humanities, social sciences, and more. We focused on three legally-relevant areas for our evaluation.

4. **Implementation Details**: Our implementation uses the DeepEval framework with 5-shot examples (`n_shots=5`). The model is instructed to output only the letter corresponding to the correct answer. For example, a prompt might look like:

   ```
   Question: In which case did the Supreme Court establish the "separate but equal" doctrine?
   A. Plessy v. Ferguson
   B. Brown v. Board of Education
   C. Dred Scott v. Sandford
   D. Korematsu v. United States

   Answer:
   ```

5. **Scoring**: Performance is measured as simple accuracy – the percentage of questions answered correctly. Since there are four options per question, random guessing would yield approximately 25% accuracy.

The overall MMLU scores from our benchmark summary are shown below:

| Model | Overall MMLU Score |
|-------|-------------------|
| Base (TinyLlama 1.1B) | 0.2472934473 |
| Fine-tuned | 0.2410256410 |

While these scores may appear modest, they're reasonable for a specialized 1.1B parameter model, as the legal and ethical reasoning tasks in MMLU are challenging even for much larger models.

## Model Performance Comparison

After completing the training and influence analysis pipeline, we evaluated our model's performance using two key metrics: BERT-Score and Semantic Similarity. The chart below shows a comparison between the base TinyLlama model and our fine-tuned version.

![Model Performance Comparison](results/combined/model_scores_comparison.png)

### Understanding the Evaluation Metrics

#### BERT-Score

BERT-Score is a text generation evaluation metric that uses the contextual embeddings from BERT to measure the semantic similarity between generated text and a reference. Unlike traditional metrics like BLEU or ROUGE that rely on exact n-gram matching, BERT-Score captures meaning at a deeper level by:

1. **Embedding Generation**: Both the candidate (model-generated) text and reference text are tokenized and processed through a pre-trained BERT model to obtain contextual embeddings for each token.
2. **Token Matching**: Rather than requiring exact matches, BERT-Score computes cosine similarity between each token in the candidate text and each token in the reference text.
3. **Greedy Matching**: It uses a greedy approach to align tokens from the candidate with the most similar tokens in the reference.
4. **F1 Calculation**: The final score combines precision (how much of the candidate text is in the reference) and recall (how much of the reference is covered by the candidate) into an F1 score.

This approach effectively captures paraphrasing, synonyms, and other semantic variations that traditional metrics might miss. In our implementation, we used the `bert-score` package, leveraging its default F1 computation with the English language model.

#### Semantic Similarity Score

The Semantic Similarity metric provides a more holistic measure of overall semantic equivalence between texts. Unlike BERT-Score, which operates on token-level alignments, this metric:

1. **Full Text Embedding**: Uses a sentence transformer model (specifically `all-MiniLM-L6-v2`) to encode entire texts into single, fixed-length vector representations.
2. **Cosine Similarity**: Calculates the cosine similarity between these sentence embeddings, which measures the angle between the vectors in a high-dimensional space.
3. **Multiple References**: When multiple reference completions are available for a prompt, it calculates similarity against each one and takes the highest score.

The advantage of this approach is that it captures the overall meaning and intent of the entire text, even when the wording or structure differs significantly. It's particularly useful for evaluating whether the model captures the core concepts required in the answer, regardless of the specific phrasing used.

### Interpreting the Results

As shown in the chart, the fine-tuned model outperforms the base model on both metrics:

- **BERT-Score**: Increased from approximately 0.81 to 0.87, representing around a 7% improvement. This indicates that our fine-tuned model produces text that is semantically closer to the reference answers at the token level.
- **Semantic Similarity**: Showed a more dramatic improvement from about 0.38 to 0.58, a roughly 53% increase. This suggests the fine-tuned model produces answers that are substantially more aligned with the expected responses in terms of overall meaning.

The more significant gain in Semantic Similarity compared to BERT-Score suggests that while both models improve in token-level semantic matching, the fine-tuning process particularly enhanced the model's ability to capture the overall meaning and intent required by the patent classification tasks, even when the exact phrasing differs.

## Attribution Analysis: Linking Outputs to Training Data

One of the most valuable aspects of our approach is the ability to trace model outputs back to specific training examples that influenced them. This attribution capability provides transparency into the model's decision-making process and helps identify which training examples had the greatest impact on particular predictions.

### Example 1: Patent Chemistry Query

**Prompt:** "What are some examples of isocyanate-reactive compounds used in the production of tin-free polyurethane foam?"

**Generated Completion:** "Examples of iso
High density fiberboard consists essentially of fiber with a diameter ranging from 0.1 to 4 mm, which are bonded together by means of isophorone diisocyanates and optionally further crosslinking agents."

**Most Influential Training Examples:**

| Rank | Influence Score | Training Example Question | Training Example Answer |
|------|----------------|--------------------------|------------------------|
| 1 | 3,244,032 | What are some examples of isocyanate-reactive compounds used in the production of tin-free polyurethane foam? | Examples include polyols (polyether polyols, polyester polyols), polyamines, and hydroxyl-terminated polybutadiene. Water also acts as an isocyanate-reactive compound in foam production, generating carbon dioxide as a blowing agent. |
| 2 | 1,662,976 | What are the advantages of using alkali metal carboxylates as catalysts in the production of tin-free polyurethane foam? | Alkali metal carboxylates provide several advantages: they're environmentally friendly alternatives to tin catalysts, exhibit strong gelation catalysis, produce foams with excellent physical properties, reduce emissions of volatile organic compounds, and comply with regulatory requirements limiting heavy metals in consumer products. |
| 3 | 1,589,248 | What is the role of alkali metal carboxylates in the production of tin-free polyurethane foam? | Alkali metal carboxylates serve as catalysts that promote the reaction between isocyanates and polyols, specifically accelerating the gelation reaction. They help maintain proper balance between gelation and blowing reactions, ensuring foam with uniform cell structure and appropriate mechanical properties. |

This complete view of the QA pairs reveals important information: while the model's response references "isophorone diisocyanates" (which relates to isocyanates mentioned in the training data), it doesn't fully capture the specific compounds listed in the most influential training example. Instead, it appears to conflate information about polyurethane foam with information about fiberboard applications. This demonstrates how attribution analysis can help identify potential confusion in the model's knowledge integration.

### Example 2: Construction Materials Query

**Prompt:** "What is the composition of the acrylic copolymer used in the multilayer composite and underlayment for roof assembly?"

**Generated Completion:** "The acrylic copolysurface is a combination of at least one (meth)acrylate component, which can be either an emulsion or a solution, along with a polyisocyanate component."

**Most Influential Training Examples:**

| Rank | Influence Score | Training Example Question | Training Example Answer |
|------|----------------|--------------------------|------------------------|
| 1 | 2,156,544 | What is the composition of the acrylic copolymer used in the multilayer composite and underlayment for roof assembly? | The acrylic copolymer consists of a combination of methyl methacrylate (MMA), butyl acrylate (BA), and acrylic acid in a specific ratio, typically 45-55% MMA, 40-50% BA, and 1-5% acrylic acid. It's formulated as a water-based emulsion with a solid content of approximately 50-60%. |
| 2 | 1,198,336 | What additives can be incorporated into the acrylic copolymer to enhance its performance in roofing applications? | Common additives include UV stabilizers to prevent degradation from sunlight, fungicides to prevent mold growth, plasticizers to improve flexibility in cold temperatures, flame retardants to meet fire codes, and silane coupling agents to improve adhesion to various substrates. |
| 3 | 982,464 | How is the acrylic copolymer applied in the multilayer roof assembly? | The acrylic copolymer is typically applied as a liquid waterproofing membrane using spray equipment, rollers, or brushes. It's applied at a thickness of 20-30 mils (0.5-0.75mm) and forms a seamless, elastomeric membrane upon curing, which typically takes 24-48 hours depending on temperature and humidity conditions. |

With the complete QA pairs, we can see that while the model's answer incorporates some accurate elements (mentioning "(meth)acrylate component" and emulsion), it also introduces a potentially incorrect association with polyisocyanates, which weren't mentioned in the top influential training examples. This could be the result of interference from other training examples or an attempt to synthesize information from multiple sources.

### Value of Attribution Analysis

This attribution capability offers several benefits:

1. **Transparency**: We can explain why the model provides particular answers by identifying the source training examples.
2. **Quality Control**: By examining influential examples, we can identify and correct problematic or inaccurate training data.
3. **Targeted Improvement**: Understanding which examples have the greatest influence allows for more focused dataset enhancement.
4. **Debugging**: When the model produces incorrect outputs, we can trace back to the source of the error in the training data.
