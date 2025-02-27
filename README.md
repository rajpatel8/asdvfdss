# COMP-8730 Natural Language Processing & Understanding
## Midterm Exam Solutions

### Q1: [20 marks]
Given the paragraph and the test sentence $l$ = "corpus is the plural form of corpora"

**a) Calculate the exact negative loglikelihood (nll) of $l$ using an n-gram LM where $n = |l|$.**

First, let's identify the tokens in the sentence $l$:
$l$ = ["corpus", "is", "the", "plural", "form", "of", "corpora"]

For an n-gram language model where n = |l| = 7 (the number of tokens in the sentence), the exact negative log-likelihood would consider the entire sentence as a single sequence.

Since we're using the paragraph as our corpus for training, we need to check if the exact 7-gram "corpus is the plural form of corpora" appears in the paragraph. Looking at the provided paragraph, this exact sequence does not appear.

For an n-gram of length |l|, if the exact sequence doesn't appear in the training data, the probability is effectively 0, and the negative log-likelihood would be:
$-\log(0) = \infty$

Therefore, the exact negative log-likelihood using a 7-gram LM is $\infty$, indicating that this specific sequence was never observed in our training data.

**b) Approximate nll of $l$ using bi-gram and tri-gram LMs using chain rule of probability and/or any smoothing techniques.**

Let's use Laplace (add-1) smoothing for our calculations. First, I'll tokenize the paragraph:

Paragraph tokens: ["In", "linguistics", "a", "corpus", "plural", "corpora", "or", "text", "corpus", "is", "a", "language", "resource", "It", "consists", "of", "a", "large", "and", "structured", "set", "of", "texts", "Nowadays", "it", "is", "usually", "electronically", "stored", "and", "processed", "In", "corpus", "linguistics", "they", "are", "used", "to", "do", "a", "set", "of", "statistical", "analysis", "and", "hypothesis", "testing", "They", "are", "also", "used", "for", "checking", "occurrences", "or", "validating", "the", "linguistic", "rules", "within", "a", "specific", "language", "territory"]

Let me calculate unique vocabulary:
V = {"In", "linguistics", "a", "corpus", "plural", "corpora", "or", "text", "is", "language", "resource", "It", "consists", "of", "large", "and", "structured", "set", "texts", "Nowadays", "it", "usually", "electronically", "stored", "processed", "they", "are", "used", "to", "do", "statistical", "analysis", "hypothesis", "testing", "They", "also", "for", "checking", "occurrences", "validating", "the", "linguistic", "rules", "within", "specific", "territory", "form"}

Note: I added "form" since it appears in our test sentence but not in the paragraph.

Let's denote |V| as the vocabulary size.
|V| = 47

Test sentence: ["corpus", "is", "the", "plural", "form", "of", "corpora"]

**Bigram Calculation (n=2)**:

Using chain rule: P(w₁, w₂, ..., wₙ) = P(w₁) × P(w₂|w₁) × P(w₃|w₂) × ... × P(wₙ|wₙ₋₁)

For a bigram model with add-1 smoothing:
P(w₁) = (count(w₁) + 1) / (N + |V|)
P(wᵢ|wᵢ₋₁) = (count(wᵢ₋₁, wᵢ) + 1) / (count(wᵢ₋₁) + |V|)

Where N is the total number of tokens in the paragraph (68).

For our sentence:
1. P("corpus") = (3 + 1) / (68 + 47) = 4/115
2. P("is"|"corpus") = (1 + 1) / (3 + 47) = 2/50
3. P("the"|"is") = (0 + 1) / (2 + 47) = 1/49
4. P("plural"|"the") = (0 + 1) / (1 + 47) = 1/48
5. P("form"|"plural") = (0 + 1) / (1 + 47) = 1/48
6. P("of"|"form") = (0 + 1) / (0 + 47) = 1/47
7. P("corpora"|"of") = (0 + 1) / (4 + 47) = 1/51

The negative log-likelihood is:
nll = -log(4/115 × 2/50 × 1/49 × 1/48 × 1/48 × 1/47 × 1/51)
    = -log(4/115) - log(2/50) - log(1/49) - log(1/48) - log(1/48) - log(1/47) - log(1/51)
    ≈ 3.36 + 3.22 + 3.89 + 3.87 + 3.87 + 3.85 + 3.93
    ≈ 25.99

**Trigram Calculation (n=3)**:

For a trigram model with add-1 smoothing:
P(w₁) and P(w₂|w₁) same as bigram
P(wᵢ|wᵢ₋₂, wᵢ₋₁) = (count(wᵢ₋₂, wᵢ₋₁, wᵢ) + 1) / (count(wᵢ₋₂, wᵢ₋₁) + |V|)

1. P("corpus") = 4/115 (same as bigram)
2. P("is"|"corpus") = 2/50 (same as bigram)
3. P("the"|"corpus", "is") = (0 + 1) / (1 + 47) = 1/48
4. P("plural"|"is", "the") = (0 + 1) / (0 + 47) = 1/47
5. P("form"|"the", "plural") = (0 + 1) / (0 + 47) = 1/47
6. P("of"|"plural", "form") = (0 + 1) / (0 + 47) = 1/47
7. P("corpora"|"form", "of") = (0 + 1) / (0 + 47) = 1/47

The negative log-likelihood is:
nll = -log(4/115 × 2/50 × 1/48 × 1/47 × 1/47 × 1/47 × 1/47)
    = -log(4/115) - log(2/50) - log(1/48) - log(1/47) - log(1/47) - log(1/47) - log(1/47)
    ≈ 3.36 + 3.22 + 3.87 + 3.85 + 3.85 + 3.85 + 3.85
    ≈ 25.85

**c) Compare above LMs in terms of effectiveness.**

Let's compare the three models:
- 7-gram (n=|l|): nll = ∞ (because the exact sequence doesn't appear in the training data)
- Bigram (n=2): nll ≈ 25.99
- Trigram (n=3): nll ≈ 25.85

From these results, we can observe:

1. The 7-gram model is extremely ineffective because it requires the exact sequence of 7 words to appear in the training data, which is highly improbable with limited data.

2. The trigram model has a slightly lower nll than the bigram model, indicating it's slightly more effective. This is because the trigram model can capture more context than the bigram model.

3. However, the difference between bigram and trigram is small, suggesting that with this limited training data, the additional context provided by the trigram doesn't offer much improvement.

4. Both the bigram and trigram models benefit greatly from smoothing, which allows them to assign non-zero probabilities to unseen sequences.

In general, with limited training data, simpler models (like bigram) may be more robust, while with more data, more complex models (like trigram or higher) can capture more nuanced patterns. In this specific case, the trigram model shows a marginal improvement over the bigram model.

### Q2: [25 marks]
Given:
- English tweets: 51%
- Japanese tweets: 20%
- In English: 30% women, 50% men
- In Japanese: 60% women, 30% men

**a) We observe a tweet in Japanese, what would be the author's gender?**

We need to find P(Gender | Japanese).

Using Bayes' rule:
P(Gender | Japanese) = P(Japanese | Gender) × P(Gender) / P(Japanese)

Let's calculate the probability for women:
P(Woman | Japanese) = P(Japanese | Woman) × P(Woman) / P(Japanese)

We need to calculate P(Woman) first. We know:
- P(Woman | English) = 0.3
- P(Woman | Japanese) = 0.6
- P(English) = 0.51
- P(Japanese) = 0.2

We can use the law of total probability:
P(Woman) = P(Woman | English) × P(English) + P(Woman | Japanese) × P(Japanese) + P(Woman | Other) × P(Other)

We don't know P(Woman | Other), but we can make a reasonable assumption. Let's assume that for other languages, the gender distribution follows the overall distribution. For simplicity, let's assume P(Woman | Other) = 0.5.

P(Woman) = 0.3 × 0.51 + 0.6 × 0.2 + 0.5 × (1 - 0.51 - 0.2)
P(Woman) = 0.153 + 0.12 + 0.5 × 0.29
P(Woman) = 0.273 + 0.145
P(Woman) = 0.418

Now, we can calculate:
P(Woman | Japanese) = P(Japanese | Woman) × P(Woman) / P(Japanese)

P(Japanese | Woman) = P(Woman | Japanese) × P(Japanese) / P(Woman)
P(Japanese | Woman) = 0.6 × 0.2 / 0.418
P(Japanese | Woman) = 0.12 / 0.418
P(Japanese | Woman) = 0.287

Now:
P(Woman | Japanese) = 0.287 × 0.418 / 0.2
P(Woman | Japanese) = 0.12 / 0.2
P(Woman | Japanese) = 0.6

Similarly, for men:
P(Man | Japanese) = P(Japanese | Man) × P(Man) / P(Japanese)

We can calculate P(Man) similar to how we calculated P(Woman):
P(Man) = P(Man | English) × P(English) + P(Man | Japanese) × P(Japanese) + P(Man | Other) × P(Other)
P(Man) = 0.5 × 0.51 + 0.3 × 0.2 + 0.5 × (1 - 0.51 - 0.2)
P(Man) = 0.255 + 0.06 + 0.5 × 0.29
P(Man) = 0.315 + 0.145
P(Man) = 0.46

Now:
P(Japanese | Man) = P(Man | Japanese) × P(Japanese) / P(Man)
P(Japanese | Man) = 0.3 × 0.2 / 0.46
P(Japanese | Man) = 0.06 / 0.46
P(Japanese | Man) = 0.13

Finally:
P(Man | Japanese) = 0.13 × 0.46 / 0.2
P(Man | Japanese) = 0.06 / 0.2
P(Man | Japanese) = 0.3

Given that P(Woman | Japanese) = 0.6 and P(Man | Japanese) = 0.3, the more likely gender for a Japanese tweet is a woman.

**b) A woman wants to tweet, what would be her language?**

We need to find P(Language | Woman).

Using Bayes' rule:
P(Language | Woman) = P(Woman | Language) × P(Language) / P(Woman)

For English:
P(English | Woman) = P(Woman | English) × P(English) / P(Woman)
P(English | Woman) = 0.3 × 0.51 / 0.418
P(English | Woman) = 0.153 / 0.418
P(English | Woman) ≈ 0.366 or 36.6%

For Japanese:
P(Japanese | Woman) = P(Woman | Japanese) × P(Japanese) / P(Woman)
P(Japanese | Woman) = 0.6 × 0.2 / 0.418
P(Japanese | Woman) = 0.12 / 0.418
P(Japanese | Woman) ≈ 0.287 or 28.7%

For other languages:
P(Other | Woman) = P(Woman | Other) × P(Other) / P(Woman)
P(Other | Woman) = 0.5 × 0.29 / 0.418
P(Other | Woman) = 0.145 / 0.418
P(Other | Woman) ≈ 0.347 or 34.7%

Since P(English | Woman) is the highest at 36.6%, the most likely language for a woman to tweet in is English.

**c) A person wants to tweet, what would be the language?**

This is simply asking for the most probable language without any gender condition.

Given:
- P(English) = 0.51
- P(Japanese) = 0.2
- P(Other) = 1 - 0.51 - 0.2 = 0.29

Since P(English) = 0.51 is the highest, the most likely language for a person to tweet in is English.

### Q3: [10 marks]
**Learning to RegEx**

A learning-based approach to developing regular expressions for sentence segmentation:

**Training Set:**
1. Corpus of documents with sentences already segmented (gold standard)
2. For each potential sentence boundary (e.g., periods, question marks, exclamation marks), extract a window of tokens around it
3. Label each potential boundary as a true sentence boundary (1) or not (0)

**Feature Engineering:**
1. Character n-grams around the potential boundary
2. Presence of capital letters after the potential boundary
3. Presence of quotation marks, parentheses
4. Presence of abbreviations (e.g., "Mr.", "Dr.", "Inc.")
5. Length of tokens before and after the boundary
6. POS tags of surrounding tokens

**Machine Learning Method:**
Binary classification using supervised learning:
1. Decision trees or random forests - good for interpretability and can capture complex decision boundaries
2. Gradient boosting methods (like XGBoost) - powerful and can be extracted as rules
3. Logistic regression with L1 regularization - produces sparse, interpretable rules

**Pattern Extraction:**
1. Train the model to predict whether a potential boundary is a true sentence boundary
2. For tree-based models, extract decision paths as rules
3. For logistic regression, extract features with significant weights
4. Convert these rules into regular expressions

**Evaluation:**
1. Precision, recall, and F1-score on a test set
2. Compare against manually designed RegEx rules
3. Measure performance across different domains and languages

**Advantages:**
1. Interpretability: Rules can be extracted and understood
2. Generalizability: Can adapt to different languages/domains
3. Maintainability: Can be retrained with new data
4. Simplicity: Final output is still a regular expression that can be used in production

This approach combines the best of both worlds: the adaptability and learning capability of machine learning with the interpretability and deployment simplicity of regular expressions.

### Q4: [10 marks]
**For a scam email detection, which is important, precision, recall or both?**

For scam email detection, where emails labeled as scams are deleted permanently after a week:

In this context, precision is more important than recall. Let me explain why:

**Definitions:**
- Precision = TP / (TP + FP) = proportion of correctly identified scams among all emails labeled as scams
- Recall = TP / (TP + FN) = proportion of correctly identified scams among all actual scam emails

**Importance of Precision:**
1. Emails labeled as scams are deleted permanently after a week. If we have low precision (many false positives), we risk deleting legitimate emails that were incorrectly classified as scams.
2. Users may never know that important legitimate emails were incorrectly classified and deleted, leading to serious consequences like missed business opportunities, important notifications, or personal communications.

**Consequences of Low Recall:**
1. If we focus too much on precision at the expense of recall, some scam emails might be missed.
2. However, these emails would still be in the inbox, and users might be able to identify them manually.
3. The harm of receiving a scam email (which might be ignored or identified by the user) is generally less severe than permanently losing a legitimate important email.

**Real-world Considerations:**
- For most users, false positives (legitimate emails classified as scams) are more annoying and potentially harmful than false negatives (scam emails not caught).
- The permanent deletion aspect significantly raises the importance of precision.
- A balanced approach might include:
  - High precision classifier for automatic deletion
  - Secondary lower-threshold detection that marks suspicious emails but doesn't delete them
  - User education on identifying scams

**Conclusion:**
Given that emails labeled as scams are permanently deleted after a week, precision should be prioritized over recall in this specific scenario. The cost of false positives (deleting legitimate emails) outweighs the cost of false negatives (missing some scam emails) due to the permanent and irreversible nature of the deletion.

### Q5: [20 marks]
**Challenges of evaluating a LM based on a Bayesian classifier using classification metrics:**

Evaluating a language model (LM) based on a Bayesian classifier using classification metrics presents several challenges:

**1. Probabilistic Nature vs. Discrete Decisions:**
- Language models typically output probability distributions over words or sequences
- Classification metrics like precision and recall require discrete decisions (class assignments)
- Converting probabilistic outputs to binary decisions requires thresholding, which introduces another hyperparameter

**2. Evaluation of Conditional Probabilities:**
- Bayesian classifiers estimate P(class|text) using the language model's P(text|class)
- LMs are evaluated on how well they model P(text|class), not directly on classification performance
- The relation between good language modeling (perplexity) and good classification is not always straightforward

**3. Class Imbalance Issues:**
- Many text classification tasks have imbalanced class distributions
- Precision and recall can be misleading with imbalanced data
- A language model might have excellent perplexity but still perform poorly on rare classes

**4. Contextual and Semantic Understanding:**
- Classification metrics don't capture whether the model understands the semantic meaning
- A model might classify correctly based on superficial patterns rather than true understanding
- Precision/recall don't evaluate the quality of the language representation

**5. Generative vs. Discriminative Evaluation:**
- LMs are generative models while classification metrics evaluate discriminative performance
- A model that generates realistic text within a class might still be poor at discriminating between classes
- The goals of generation and discrimination can sometimes conflict

**6. Out-of-Vocabulary and Novel Inputs:**
- LMs have a fixed vocabulary and may struggle with unseen words
- Classification metrics don't specifically address how the model handles novel inputs
- Handling OOV words differently can significantly impact classification performance without changing the LM's intrinsic quality

**7. Interpretability Challenges:**
- When a Bayesian classifier makes an error, it's difficult to determine if the fault lies in:
  - The language model component (P(text|class))
  - The prior probabilities (P(class))
  - The approximations made in the Bayesian framework
- Precision and recall don't provide insight into which component needs improvement

**8. Smoothing Effects:**
- Bayesian classifiers typically use smoothing to handle unseen events
- Different smoothing methods affect classification metrics differently
- The choice of smoothing method becomes an additional factor in evaluation

**9. Document Length Sensitivity:**
- Precision and recall treat all documents equally regardless of length
- The probability estimates from language models are sensitive to document length
- Long documents might dominate the evaluation even if they're less representative

**10. Calibration Issues:**
- Language models might not be well-calibrated (probabilities don't match empirical frequencies)
- A poorly calibrated LM can lead to poor classification despite modeling the data distribution well
- Precision and recall don't assess probability calibration

These challenges highlight why evaluating language models using classification metrics requires careful consideration of the model's purpose, the task requirements, and appropriate evaluation frameworks that might extend beyond simple precision and recall.

### Q6: [15 marks]
**Given a user's timeline in a social network platform such as Facebook, we want to classify a particular post of the user, at time t, into a predefined set of {'happy', 'sad', 'meh'}. Glory suggests using a bidirectional LM to consider all posts after t and before t. What would be her reasons? Do you agree? Explain your answer.**

Glory's suggestion to use a bidirectional LM for emotion classification in social media has several potential reasons:

**Potential Reasons for Glory's Suggestion:**

1. **Contextual Understanding:** 
   - Posts before time t establish the user's baseline emotional state and ongoing life events
   - Posts after time t may reveal the emotional consequences or outcomes of events mentioned at time t
   - Together, they provide a fuller contextual understanding

2. **Emotional Continuity:**
   - Emotions rarely exist in isolation; they follow patterns and progressions
   - A bidirectional approach can capture emotional arcs and trajectories
   - For example, a "sad" post might be better understood in the context of happy posts before and after (suggesting a temporary setback)

3. **Ambiguity Resolution:**
   - Many posts contain ambiguous language or mixed emotions
   - Future posts may clarify the intended sentiment of earlier posts
   - Previous posts may establish patterns of expression specific to that user

4. **Long-term Emotional Context:**
   - Users develop personal emotional baselines and expression styles
   - A "happy" post for one user might use language that appears "meh" for another user
   - Bidirectional context helps establish these personal patterns

5. **Response Incorporation:**
   - Posts after time t might include reactions to responses the post received
   - These can clarify the original emotional intent or show how it evolved

**My Assessment:**

I partially agree with Glory's suggestion, with important caveats:

**Advantages of the Bidirectional Approach:**
- It provides richer contextual information for classification
- It can capture emotional progressions and personal expression patterns
- It aligns with how humans understand emotions in social contexts

**Disadvantages and Concerns:**
1. **Causality Violation:**
   - Using future posts to classify current posts creates a causality problem
   - In real-time applications, future posts wouldn't be available at classification time
   - This limits practical applicability to retrospective analysis

2. **Temporal Relevance Decay:**
   - Posts far in the future/past may have little relevance to the current emotional state
   - Including them could introduce noise rather than signal
   - A weighted approach giving more importance to temporally close posts would be better

3. **Privacy and Ethical Considerations:**
   - Analyzing a broader timeline increases the privacy footprint of the system
   - Users might consent to analysis of individual posts but not their entire timeline

4. **Computational Efficiency:**
   - Processing entire timelines is computationally expensive
   - The gain in accuracy might not justify the increased computational cost

**Alternative Approach:**
A more balanced approach might be:
- Use bidirectional LMs for research and understanding emotional patterns
- Implement a forward-only model (posts before time t) for real-time applications
- Add a limited "look-ahead" window for delayed classification when appropriate
- Incorporate user-specific emotional baselines learned over time

**Conclusion:**
While Glory's suggestion has merit from a modeling perspective, practical constraints around causality and computational efficiency suggest a more nuanced implementation. The ideal approach would leverage the benefits of bidirectional context while respecting the temporal nature of social media posts and the constraints of real-world applications.
