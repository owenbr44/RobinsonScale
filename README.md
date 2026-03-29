# Robinson Scale

A statistically grounded rating system that replaces traditional 0–10 scales with a **-5 to 5 scale centered at true neutrality (0)** and modeled as a **normal distribution**.

---

## Overview

Most rating systems are flawed because they become **inflated** — people tend to rate things as 7, 8, or 9 even when they are average.

The **Robinson Scale** corrects this by:

- Centering the scale at **0 (true neutral)**
- Using a range of **-5 to 5**
- Assuming evaluations follow a **normal distribution**
- Making extreme values **rare and meaningful**

---

## Key Idea

> Most things are average.

- **0** = expected, unremarkable, baseline  
- **Positive values** = above average  
- **Negative values** = below average  

---

## Statistical Foundation

The Robinson Scale assumes a normal distribution:

- Mean (μ) = 0  
- Standard deviation (σ) ≈ 1.67  
- Range (-5 to 5) ≈ ±3σ  

This implies:

| Range | Approx % | Meaning |
|------|--------|--------|
| -1 to 1 | ~46% | Most experiences |
| -2 to 2 | ~76% | Common variation |
| -3 to 3 | ~92% | Strong but not rare |
| -5 to 5 | ~99.7% | Full range |
| ±5 | ~0.3% total | Extreme outcomes |

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

- Stores rated items in a `things.csv` file  
- Visualizes the Robinson Scale as a **normal distribution curve**  
- Displays:
  - Probability percentages per region  
  - Count of items at each score (`n=`)  
  - One example per score (`ex:`)  
- Automatically saves a **high-resolution PNG graph**  

---

## Installation


Install required packages:

```bash
pip install numpy matplotlib


