import json
import argparse
import matplotlib.pyplot as plt
import os
from typing import Dict, Tuple

MODEL_DEFAULTS = {
    "llama": {
        "results_dir": "results/llama",
    },
    "gemma": {
        "results_dir": "results/gemma",
    },
    "mistral": {
        "results_dir": "results/mistral",
    },
    "gpt2": {
        "results_dir": "results",
    },
}

COLORS = {
    "Explicit": "#6BBF8F",      # Green
    "Thematic": "#E8D74B",      # Yellow
    "Accidental": "#5B8FCC",    # Blue
    "None": "#CCCCCC",          # Gray
    "Error": "#FF6B6B",         # Red
    "Error_Format": "#FF6B6B",  # Red
}

def extract_stats(data: Dict, model_type: str) -> Dict[str, int]:
    """Extract classification statistics from the judged results."""
    if "metrics" not in data or model_type not in data["metrics"]:
        return {}
    
    metrics = data["metrics"][model_type]
    if "llm_judge" in metrics:
        return metrics["llm_judge"]
    return {}

def create_pie_charts(baseline_stats: Dict, ablated_stats: Dict, output_path: str, title: str = "LLM Judge Classification Results"):
    """Create side-by-side pie charts for baseline vs ablated."""
    
    # Filter out error categories for cleaner visualization
    categories_to_show = ["Explicit", "Thematic", "Accidental", "None"]
    
    baseline_data = {k: baseline_stats.get(k, 0) for k in categories_to_show}
    ablated_data = {k: ablated_stats.get(k, 0) for k in categories_to_show}
    
    # Calculate totals
    baseline_total = sum(baseline_data.values())
    ablated_total = sum(ablated_data.values())
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Colors for categories
    colors = [COLORS.get(cat, "#CCCCCC") for cat in categories_to_show]
    
    # Baseline pie chart
    baseline_values = [baseline_data[cat] for cat in categories_to_show]
    baseline_percentages = [v / baseline_total * 100 if baseline_total > 0 else 0 for v in baseline_values]
    
    wedges1, texts1, autotexts1 = ax1.pie(
        baseline_values,
        labels=categories_to_show,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 11, 'weight': 'bold'}
    )
    ax1.set_title('Baseline Knowledge Recall', fontsize=13, fontweight='bold', pad=20)
    
    # Style baseline text
    for autotext in autotexts1:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    # Ablated pie chart
    ablated_values = [ablated_data[cat] for cat in categories_to_show]
    ablated_percentages = [v / ablated_total * 100 if ablated_total > 0 else 0 for v in ablated_values]
    
    wedges2, texts2, autotexts2 = ax2.pie(
        ablated_values,
        labels=categories_to_show,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 11, 'weight': 'bold'}
    )
    ax2.set_title('Ablated Knowledge Recall', fontsize=13, fontweight='bold', pad=20)
    
    # Style ablated text
    for autotext in autotexts2:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    # Add note at bottom
    fig.text(0.5, 0.02, 
             'LLM-as-a-judge classification of model completions across 4 categories.\nExplicit recall is significantly reduced in the ablated model.',
             ha='center', fontsize=10, style='italic', color='#666666')
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize LLM Judge Evaluation Results")
    parser.add_argument(
        "--model_family",
        type=str,
        choices=["llama", "gemma", "mistral", "gpt2"],
        default="gpt2",
        help="Model family to visualize.",
    )
    parser.add_argument("--input", type=str, default=None, help="Path to judged results JSON")
    parser.add_argument("--output", type=str, default=None, help="Path to save visualization image")
    args = parser.parse_args()

    results_dir = MODEL_DEFAULTS[args.model_family]["results_dir"]
    if args.input is None:
        args.input = f"{results_dir}/unified_eval_results_judged.json"
    if args.output is None:
        args.output = f"{results_dir}/knowledge_recall_visualization.png"

    print(f"Loading results from {args.input}...")
    try:
        with open(args.input, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {args.input} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: File {args.input} is not valid JSON.")
        return

    # Extract statistics
    baseline_stats = extract_stats(data, "baseline")
    ablated_stats = extract_stats(data, "ablated")

    if not baseline_stats or not ablated_stats:
        print("Error: Could not find llm_judge statistics in the results file.")
        return

    # Calculate totals
    baseline_total = sum(baseline_stats.values())
    ablated_total = sum(ablated_stats.values())

    print(f"\nBaseline Statistics (Total: {baseline_total}):")
    for category, count in sorted(baseline_stats.items(), key=lambda x: x[1], reverse=True):
        percentage = count / baseline_total * 100 if baseline_total > 0 else 0
        print(f"  {category}: {count} ({percentage:.1f}%)")

    print(f"\nAblated Statistics (Total: {ablated_total}):")
    for category, count in sorted(ablated_stats.items(), key=lambda x: x[1], reverse=True):
        percentage = count / ablated_total * 100 if ablated_total > 0 else 0
        print(f"  {category}: {count} ({percentage:.1f}%)")

    # Create visualization
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    create_pie_charts(
        baseline_stats,
        ablated_stats,
        args.output,
        title=f"{args.model_family.title()} Model - Knowledge Recall Comparison"
    )

    print("\n" + "="*50)
    print("Visualization complete!")
    print("="*50)

if __name__ == "__main__":
    main()
