import json
import glob
import pandas as pd
import matplotlib.pyplot as plt

def show_results(root_path, tp):
    files = glob.glob(f"{root_path}/**/*summary.json", recursive=True)
    print(f"Found {len(files)} files")

    data = []

    for file in files:
        with open(file, 'r') as f:
            json_data = json.load(f)
            json_data['file'] = file
            json_data['tp'] = int(json_data['tp'])

            data.append(json_data)

    df = pd.DataFrame(data)
    df = df.sort_values(['engine', 'tp', 'dtype', 'num_concurrent_requests'])

    plt.figure(figsize=(16, 12))

    tp_size = [tp]

    metrics = [
        "results_request_output_throughput_token_per_s_mean",
        "results_ttft_s_quantiles_p95"
    ]

    titles = [
        "Mean throughput vs Concurrency",
        "TTFT (p95) vs Concurrency"
    ]

    y_labels = [
        "Throughput (tokens/s)",
        "TTFT (s)"
    ]

    thresholds = {
        "results_request_output_throughput_token_per_s_mean": 40,
        "results_ttft_s_quantiles_p95": 0.2
    }

    for i, (metric, title, y_label) in enumerate(zip(metrics, titles, y_labels), 1):
        plt.subplot(1, 2, i)

        for engine in df["engine"].unique():
            subset = df.query("tp in @tp_size and engine == @engine").copy()
            subset_bf16 = subset.query("dtype=='bf16'")
            subset_fp8 = subset.query("dtype=='fp8'")

            if len(subset_bf16):
                bf16_line, = plt.plot(subset_bf16["num_concurrent_requests"],
                                    subset_bf16[metric], marker='o', label=f"engine={engine}, dtype=bf16")

            if len(subset_fp8):
                fp8_line, = plt.plot(subset_fp8["num_concurrent_requests"],
                                    subset_fp8[metric], marker='o', label=f"engine={engine}, dtype=fp8")

        # Add horizontal threshold lines
        if metric in thresholds:
            plt.axhline(y=thresholds[metric], color='r', linestyle='--',
                        label=f"Threshold: {thresholds[metric]}" + (" tokens/s" if metric == metrics[0] else " s"))

        plt.xlabel("Number of Concurrent Requests")
        plt.ylabel(y_label)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.title(f"TP={tp_size}, {title}")

        if metric in ['results_ttft_s_quantiles_p95']:
            plt.yscale("log")

        plt.xscale("log")
        plt.legend(title="Engine and dtype")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    """
    python analysis.py --root_path output_gemma4 --tp 1
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--tp", type=int, default=1)

    args = parser.parse_args()

    show_results(args.root_path, args.tp)
