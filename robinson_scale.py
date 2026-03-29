import csv
import os
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from math import erf, sqrt, pi


class RobinsonScale:
    def __init__(self, csv_file="things.csv"):
        self.csv_file = csv_file
        self.mean = 0
        self.std = 5 / 3
        self.items = []
        self.load_items()

    def pdf(self, x):
        return (1 / (self.std * sqrt(2 * pi))) * np.exp(
            -0.5 * ((x - self.mean) / self.std) ** 2
        )

    def cdf(self, x):
        return 0.5 * (1 + erf((x - self.mean) / (self.std * sqrt(2))))

    def wrap_label(self, text, width=18):
        return "\n".join(textwrap.wrap(text, width=width))

    def load_items(self):
        self.items = []

        if not os.path.exists(self.csv_file):
            with open(self.csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["name", "score"])
            return

        with open(self.csv_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    name = row["name"].strip()
                    score = float(row["score"])
                    if -5 <= score <= 5:
                        self.items.append({"name": name, "score": score})
                except Exception:
                    continue

    def save_item(self, name, score):
        with open(self.csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([name, score])

    def add_item(self, name, score):
        if not isinstance(score, (int, float)):
            raise TypeError("Score must be a number.")
        if score < -5 or score > 5:
            raise ValueError("Score must be between -5 and 5.")

        self.items.append({"name": name, "score": float(score)})
        self.save_item(name, score)
        print(f'Added "{name}" with score {score}.')

    def show_items(self):
        if not self.items:
            print("No items found.")
            return

        sorted_items = sorted(self.items, key=lambda x: x["score"], reverse=True)

        print("\nRobinson Scale Rankings")
        print("-" * 40)
        for item in sorted_items:
            print(f"{item['name']}: {item['score']}")
        print("-" * 40)

    def plot_distribution(self, save_path="robinson_scale.png"):
        plt.close("all")

        x = np.linspace(-5, 5, 1200)
        y = self.pdf(x)

        fig, ax = plt.subplots(figsize=(16, 9), dpi=200)
        ax.plot(x, y, linewidth=3, label="Robinson Scale")
        ax.axvline(0, linestyle="--", linewidth=2, label="Neutral (0)")

        # Shade score regions and label probabilities
        bounds = np.arange(-5, 6, 1)
        for i in range(len(bounds) - 1):
            a = bounds[i]
            b = bounds[i + 1]
            x_fill = np.linspace(a, b, 200)
            y_fill = self.pdf(x_fill)
            ax.fill_between(x_fill, y_fill, alpha=0.15)

            prob = self.cdf(b) - self.cdf(a)
            mid = (a + b) / 2
            height = self.pdf(mid)
            ax.text(
                mid,
                height + 0.008,
                f"{prob * 100:.1f}%",
                ha="center",
                fontsize=12,
                fontweight="bold"
            )

        # Light guide lines
        for s in range(-5, 6):
            ax.axvline(s, linestyle=":", linewidth=0.8, alpha=0.35)

        # Group items by rounded integer score
        score_groups = {}
        for item in self.items:
            score = int(round(item["score"]))
            score_groups.setdefault(score, []).append(item["name"])

        rng = np.random.default_rng()

        # Plot points and counts
        for score, names in score_groups.items():
            base_y = self.pdf(score)

            jitters = rng.uniform(-0.01, 0.01, size=len(names))
            y_vals = np.maximum(base_y + jitters, 0.002)

            ax.scatter([score] * len(names), y_vals, s=40, alpha=0.7, zorder=3)
            ax.text(score, 0.010, f"n={len(names)}", ha="center", fontsize=11)

        # Example label positions chosen to reduce overlap
        label_positions = {
            -5: (-4.7, 0.045),
            -4: (-4.0, 0.070),
            -3: (-3.25, 0.045),
            -2: (-2.25, 0.070),
            -1: (-1.05, 0.055),
             0: (0.00, 0.035),
             1: (1.05, 0.055),
             2: (2.25, 0.070),
             3: (3.25, 0.045),
             4: (4.00, 0.070),
             5: (4.75, 0.045),
        }

        # Add one example label per score
        for score, names in score_groups.items():
            example = rng.choice(names)
            label = "ex: " + self.wrap_label(example, width=16)

            x_text, y_text = label_positions.get(score, (score, 0.05))
            x_point = score
            y_point = max(self.pdf(score) * 0.25, 0.012)

            ax.annotate(
                label,
                xy=(x_point, y_point),
                xytext=(x_text, y_text),
                textcoords="data",
                ha="center",
                va="center",
                fontsize=10,
                arrowprops=dict(
                    arrowstyle="-",
                    lw=0.8,
                    alpha=0.6
                )
            )

        ax.set_title("Robinson Scale Distribution", fontsize=22, pad=20)
        ax.set_xlabel("Score", fontsize=16)
        ax.set_ylabel("Density", fontsize=16)
        ax.set_xticks(range(-5, 6))
        ax.set_xlim(-5.5, 5.5)
        ax.set_ylim(-0.005, 0.27)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=12)

        plt.subplots_adjust(bottom=0.12)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"Saved graph to: {save_path}")


if __name__ == "__main__":
    scale = RobinsonScale("things.csv")
    scale.show_items()
    scale.plot_distribution()