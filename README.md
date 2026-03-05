![spam-filter](https://github.com/njgeorge000158/Spam-Detection-with-Supervised-Machine-Learning-Models-Using-Scikit-Learn/assets/137228821/3d609d19-8f64-45ac-ab76-df20bb29db4b)

----

# **ISP Email Spam Detection: A Comparative Evaluation of Classification Models and Resampling Strategies**

---

## **Project Overview**

Email spam filtering is one of the most practically consequential binary classification problems in everyday computing. For an Internet Service Provider operating at scale, the cost of misclassification runs in both directions: spam that reaches an inbox erodes user trust and exposes customers to phishing and fraud, while legitimate email incorrectly quarantined disrupts communication and undermines the system's credibility. Building the most accurate, precise, and reliable classifier possible is therefore not merely an academic exercise — it is a direct service quality imperative.

This analysis evaluates five supervised machine learning classifiers — Logistic Regression, Random Forest, Decision Tree, Support Vector Machine (SVM), and K-Nearest Neighbor (KNN), alongside Gaussian Naive Bayes (GNB) — across six sampling methodologies, producing a comprehensive matrix of 36 optimized models trained on a dataset of token frequency features extracted from 2,788 legitimate and 1,813 spam emails. The class imbalance between normal and spam emails motivated systematic evaluation of random undersampling, random oversampling, Cluster Centroids, SMOTE, and SMOTEEN alongside baseline models.

Going into the analysis, I anticipated that Logistic Regression would outperform Random Forest, reasoning that the dataset's numerical token frequency features — continuous values measuring word occurrence rates — would align naturally with Logistic Regression's linear decision boundary. As will be shown, that prediction was incorrect in an instructive and analytically revealing way.

---

## **Methodology**

All 36 models were first optimized through an automated hyperparameter search in `spam_detector_optimization.ipynb`, with the resulting configurations saved to the `resources` folder. Each model was then loaded, fitted against the training data using its optimized parameters, and evaluated against a held-out test set of 1,151 records — 699 actual spam and 452 actual non-spam — using accuracy, precision, recall, F1-score, and confusion matrix analysis.

---

### *Logistic Regression Baseline*

<img width="432" alt="Screenshot 2024-04-17 at 3 51 23 PM" src="https://github.com/njgeorge000158/Spam-Detection-with-Supervised-Machine-Learning-Models-Using-Scikit-Learn/assets/137228821/3009deba-873d-4fec-8320-86d1e3210d41">

The Logistic Regression model achieves an overall accuracy of **89.67%** on the 1,151-record test set. The confusion matrix shows 666 true spam detections and 380 true non-spam classifications, with 33 false negatives — spam emails that slipped through to the inbox — and 72 false positives — legitimate emails incorrectly flagged as spam.

The classification report reveals an asymmetry between the two classes. Spam detection precision is 0.90 with a recall of 0.95 and an F1-score of 0.93 — the model catches nearly all spam but occasionally misidentifies legitimate email as spam. Non-spam precision is 0.92 but recall drops to 0.84, meaning 16% of legitimate emails are incorrectly quarantined — a meaningful false positive rate that would frustrate users. The macro average F1-score of 0.90 and weighted average of 0.91 confirm solid but imperfect overall performance.

The model's relatively stronger spam recall (0.95) than non-spam recall (0.84) reflects the asymmetric class distribution in the training data — the model has seen more spam examples and has learned to err on the side of flagging borderline cases as spam, at the cost of some legitimate email.

---

### *Random Forest Baseline*

<img width="431" alt="Screenshot 2024-04-17 at 3 52 02 PM" src="https://github.com/njgeorge000158/Spam-Detection-with-Supervised-Machine-Learning-Models-Using-Scikit-Learn/assets/137228821/a59bf1a5-625d-4951-9cb7-150780118771">

The Random Forest model immediately outperforms Logistic Regression with an overall accuracy of **91.35%** — a meaningful improvement of 1.68 percentage points on a test set of over 1,000 records. The confusion matrix tells a dramatically different story for false negatives: only 8 spam emails escape detection compared to 33 for Logistic Regression — a 76% reduction in missed spam. False positives remain similar at 73.

The classification report confirms the improvement. Spam recall leaps to 0.99 — near-perfect spam detection — while spam precision holds steady at 0.90. Non-spam precision improves substantially to 0.98, though non-spam recall remains at 0.84, matching Logistic Regression's performance on that class. The macro average F1-score rises to 0.92, and the weighted average to 0.93 — meaningfully better than Logistic Regression across all aggregate metrics.

The Random Forest's superior spam recall is particularly significant for an ISP deployment context. A spam filter that misses only 8 out of 699 spam emails — a miss rate of just 1.1% — while maintaining 90% spam precision represents a practically deployable system. Logistic Regression's miss rate of 4.7% is substantially less effective despite the models' superficial similarity in overall accuracy.

---

### *Random Forest with Undersampling: The Best Model*

<img width="432" alt="Screenshot 2024-04-17 at 5 08 17 PM" src="https://github.com/njgeorge000158/Spam-Detection-with-Supervised-Machine-Learning-Models-Using-Scikit-Learn/assets/137228821/2edec370-ed15-4ab6-86d8-b08a680fa4fa">

Applying random undersampling to the Random Forest classifier produces the best-performing model in the entire 36-model evaluation, achieving an overall accuracy of **93.02%**. The confusion matrix shows further improvement: 688 true spam detections (versus 691 for the baseline Random Forest — a marginal decline), but only 11 false negatives compared to 8 — a slight increase — offset by a significant reduction in false positives from 73 to 56, improving non-spam recall from 0.84 to 0.88.

The classification report for the undersampled model is the strongest across all metrics evaluated. Spam precision improves to 0.92, spam recall remains high at 0.98, and the spam F1-score reaches 0.95. Non-spam precision is 0.97, non-spam recall improves to 0.88, and the non-spam F1-score rises to 0.92. The macro average F1-score reaches 0.94 and the weighted average 0.94 — the highest values in the evaluation. The overall accuracy of 93.02% exceeds the Random Forest baseline by 1.67 percentage points and exceeds Logistic Regression by 3.35 percentage points.

The mechanism behind undersampling's benefit here is instructive. By reducing the majority class (normal emails) to match the minority class (spam) during training, the model is forced to learn a more balanced decision boundary — one that does not systematically favor the more prevalent class. The result is improved recall for the non-spam class without meaningfully sacrificing spam detection performance, producing a more symmetrically effective classifier.

---

### *Full Model Rankings (Image 4)*

<img width="347" alt="Screenshot 2024-04-17 at 3 52 44 PM" src="https://github.com/njgeorge000158/Spam-Detection-with-Supervised-Machine-Learning-Models-Using-Scikit-Learn/assets/137228821/33020891-cd3c-434e-bbd0-4cac5ffc8de5">

Table 9.7.2 presents all 36 models ranked by accuracy, and the results reveal several important patterns that extend beyond the top three models.

The top tier, spanning approximately 91.4% to 93.0%, is dominated by Random Forest and SVM configurations across multiple sampling strategies. Random Forest with undersampling (93.0%), Random Forest with SMOTEEN (92.9%), SVM with undersampling (92.8%), Random Forest with SMOTE (92.5%), and SVM with SMOTEEN (92.1%) occupy the first five positions — confirming that both Random Forest and SVM respond positively to sampling correction on this dataset and that the choice of sampling strategy meaningfully affects outcomes for these classifiers.

Logistic Regression configurations cluster in the middle tier, ranging from 89.7% to 92.0%. Logistic Regression with SMOTEEN and undersampling both reach 92.0% — substantially above the 89.7% baseline — demonstrating that resampling helps Logistic Regression too, even if the model family never reaches the heights achieved by Random Forest and SVM. The gap between the best Logistic Regression model (92.0%) and the best Random Forest model (93.0%) is modest in absolute terms but consistent across all sampling strategies, suggesting a structural performance ceiling for Logistic Regression on this data.

KNN occupies a middle position, with the baseline KNN reaching 91.9% — remarkably close to the Random Forest baseline — but KNN variants showing more variable responses to different sampling strategies, ranging from 88.7% (cluster centroids) to 92.0% (SMOTE).

Decision Tree models perform consistently below the top-tier classifiers, ranging from 62.4% (SMOTEEN — the worst result in the entire evaluation) to 89.7% (SMOTE). The dramatic variability within the Decision Tree family — a spread of over 27 percentage points — mirrors the pattern observed in the loan risk analysis, confirming that Decision Trees are highly sensitive to the choice of sampling strategy and generally less reliable for this class of binary text classification problem.

Gaussian Naive Bayes occupies the bottom tier almost exclusively, with all six GNB configurations scoring between 83.4% and 85.8%. This is a notable and somewhat counterintuitive result — GNB is a popular choice for text classification tasks due to its assumption of feature independence, which is often approximately satisfied in bag-of-words representations. The model's relatively poor performance here suggests that the token frequency features in this dataset exhibit meaningful correlations that violate the independence assumption, allowing correlation-aware models like Random Forest and SVM to learn superior decision boundaries.

---

## **Why Random Forest Outperformed Logistic Regression: A Post-Hoc Analysis**

The initial prediction that Logistic Regression would perform better proved incorrect, and understanding why is one of the analysis's most valuable takeaways.

Logistic Regression performs optimally when the relationship between the input features and the log-odds of the target class is approximately linear — that is, when the decision boundary between spam and non-spam can be drawn as a hyperplane in feature space. Token frequency features satisfy this assumption in many text classification contexts, which motivated the original prediction.

However, Random Forest's superior performance suggests that the relationship between token frequencies and spam classification in this dataset is non-linear — that spam and non-spam emails are not linearly separable, and that the interactions between specific combinations of token frequencies carry predictive information that a linear model cannot capture. Random Forest, as an ensemble of decision trees that can model arbitrary non-linear interactions and feature combinations, is better equipped to exploit these patterns. Additionally, Random Forest's ensemble averaging naturally reduces variance and overfitting on relatively small datasets like this one — a significant advantage with only 4,601 total emails in the training pool.

The lesson generalizes: the optimal classification model depends critically on the specific structure of the data, and no single algorithm can be assumed superior in advance. The opacity of real-world datasets — particularly around the presence or absence of linear separability — makes empirical evaluation across multiple model families an essential practice rather than an optional one.

---

## **Summary and Recommendations**

---

**Model Performance Summary**

| Rank | Model | Sampling | Accuracy | Spam Recall | Non-Spam Precision | F1 (Macro) |
|---|---|---|---|---|---|---|
| 1 | Random Forest | Undersampled | 93.02% | 0.98 | 0.97 | 0.94 |
| 2 | Random Forest | SMOTEEN | 92.9% | — | — | — |
| 3 | SVM | Undersampled | 92.8% | — | — | — |
| 4 | Random Forest | Baseline | 91.35% | 0.99 | 0.98 | 0.92 |
| 22 | Logistic Regression | Baseline | 89.67% | 0.95 | 0.92 | 0.90 |
| Last | Decision Tree | SMOTEEN | 62.4% | — | — | — |

---

The analysis produces a clear and deployable recommendation: **Random Forest with random undersampling** is the optimal model for the ISP's email spam filtering system, achieving 93.02% overall accuracy, 98% spam recall, 92% spam precision, and 88% non-spam recall across a balanced set of evaluation metrics.

For practical deployment, the model's near-perfect spam recall (98%) means that fewer than 2% of spam emails will reach user inboxes — an operationally strong result. Its non-spam precision of 97% means that when the model quarantines a legitimate email, it is wrong only 3% of the time, minimizing user disruption from false positives. The combination of these metrics makes the undersampled Random Forest not merely the best performer in this evaluation but a genuinely production-ready solution.

The broader finding — that a diverse empirical evaluation across multiple classifiers and sampling strategies is essential for identifying the optimal model — applies well beyond this specific dataset. The 30-percentage-point performance gap between the best and worst models in this evaluation underscores the risk of committing to any single algorithm without systematic comparison.

----

### Copyright

Nicholas J. George © 2024. All Rights Reserved.
