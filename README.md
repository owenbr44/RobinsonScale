# Robinson Scale

A statistically grounded rating system that replaces traditional 0–10 scales with a -5 to 5 scale centered at true neutrality (0) and modeled as a normal distribution.

---

## Overview

Most rating systems are flawed because they become inflated — people tend to rate things as 7, 8, or 9 even when they are average.

The Robinson Scale corrects this by:

Centering the scale at 0 (true neutral)  
Using a range of -5 to 5  
Assuming evaluations follow a normal distribution  
Making extreme values rare and meaningful  

---

## Key Idea

Most things are average.  

0 = expected, unremarkable, baseline  
Positive values = above average  
Negative values = below average  

---

## Statistical Foundation

The Robinson Scale assumes a normal distribution:

Mean (μ) = 0  
Standard deviation (σ) ≈ 1.67  
Range (-5 to 5) ≈ ±3σ  

This implies:

| Range | Approx % | Meaning |
|------|--------|--------|
| -1 to 1 | ~46% | Most experiences |
| -2 to 2 | ~76% | Common variation |
| -3 to 3 | ~92% | Strong but not rare |
| -5 to 5 | ~99.7% | Full range |
| ±5 | ~0.3% total | Extreme outcomes |

---

## Distribution in Practice

To understand how the Robinson Scale works in reality, consider 100 randomly evaluated items:

| Score | Approx Count (out of 100) |
|------|---------------------------|
| -5 | 1 |
| -4 | 2 |
| -3 | 8 |
| -2 | 16 |
| -1 | 23 |
| 0 | 24 |
| +1 | 23 |
| +2 | 16 |
| +3 | 8 |
| +4 | 2 |
| +5 | 1 |

This highlights several key insights:

Most things are neutral — about 24% fall exactly at 0  
Nearly 70% of experiences fall between -1 and +1  
Extreme values (±5) occur only about 1% of the time  

---

## Why This Matters

We often forget how many things we feel neutral about.

Traditional rating systems push us to label everything as good or bad, which leads to inflated scores. In reality:

Most experiences are simply fine — not memorable, not terrible, just expected.

The Robinson Scale restores this balance by:

Giving neutrality a true center (0)  
Making extreme ratings rare and meaningful  
Encouraging more honest and calibrated evaluations  

A score of ±5 should represent something truly exceptional or catastrophic, not just “really good” or “really bad.”

---

## Interpretation of Scores

| Score | Description |
|------|------------|
| 0 | Neutral / expected |
| ±1 | Slightly good or bad |
| ±2 | Clearly noticeable |
| ±3 | Strong, memorable |
| ±4 | Rare, exceptional |
| ±5 | Extreme, life-altering |

---

## Features

Stores rated items in a things.csv file  
Visualizes the Robinson Scale as a normal distribution curve  

Displays:  
Probability percentages per region  
Count of items at each score (n=)  
One example per score (ex:)  

Automatically saves a high-resolution PNG graph  

---

## Installation

Install required packages:

```
pip install numpy matplotlib
```

---

# Application: Recommendation Systems

One practical application of the Robinson Scale explored in this project is its use in a **movie recommendation system**.

Using the MovieLens dataset, two models were compared:

- A **standard model** trained on original 1–5 ratings  
- A **Robinson model** trained on ratings transformed into the Robinson Scale  

Both models used the same algorithm, allowing for a direct comparison of how the rating structure alone changes outcomes.

---

## Key Findings

The Robinson transformation does not improve prediction accuracy:

- Higher RMSE and MAE (worse at predicting exact ratings)

However, it changes recommendation behavior in meaningful ways:

- **Higher recall** → surfaces more items users actually like  
- **High overlap (~73%)** → preserves core recommendations  
- **Higher catalog diversity** → recommends a wider range of items across users  

This means the Robinson Scale does not simply make recommendations “more accurate,” but instead makes them **broader and more exploratory**.

---

## Interpretation

Standard recommendation systems tend to:

- favor safe, popular items  
- repeat similar suggestions  
- optimize for prediction accuracy  

The Robinson-based system instead:

- preserves strong recommendations  
- introduces more variation across users  
- increases overall system diversity  

This suggests that the Robinson Scale may be particularly useful for **discovery-focused systems**, where the goal is not just accuracy but helping users find new and relevant content.

---

## Broader Applications

The movie recommender system is just one example. The Robinson Scale can be applied anywhere ratings or evaluations exist.

### Potential use cases include:

Recommendation systems  
Reviews and rating platforms  
Surveys and public policy evaluation  
Performance reviews and grading systems  
Machine learning training data  

---

## Core Insight

The Robinson Scale is not just a different scoring system.

It is a different assumption about how evaluations work:

Most things are average  
Extremes are rare  
Neutrality should be meaningful  

By restructuring data around these principles, the Robinson Scale can improve how systems interpret and use human evaluations.

---

## Author

**Owen Robinson**  
MPA Candidate, Syracuse University
