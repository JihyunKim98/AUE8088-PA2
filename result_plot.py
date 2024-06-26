import argparse
import json
import matplotlib.pyplot as plt
import numpy as np

def evaluate(ann_file, rst_file, phase):
    # Dummy implementation of evaluation function
    # This should be replaced with actual evaluation logic
    with open(rst_file, 'r') as f:
        results = json.load(f)
    
    # Assuming results contain false positives and miss rates
    false_positives = np.array(results['false_positives'])
    miss_rate = np.array(results['miss_rate'])
    
    return false_positives, miss_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='eval models')
    parser.add_argument('--annFile', type=str, default='evaluation_script/KAIST_annotation.json',
                        help='Please put the path of the annotation file. Only support json format.')
    parser.add_argument('--rstFiles', type=str, nargs='+', default=['evaluation_script/MLPD_result.json'],
                        help='Please put the path of the result file. Only support json, txt format.')
    parser.add_argument('--evalFig', type=str, default='KASIT_BENCHMARK.jpg',
                        help='Please put the output path of the Miss rate versus false positive per-image (FPPI) curve')
    args = parser.parse_args()

    phase = "Multispectral"
    results = [evaluate(args.annFile, rstFile, phase) for rstFile in args.rstFiles]

    # Plotting
    plt.figure(figsize=(15, 5))

    titles = ['All', 'Day', 'Night']
    for i, (title, result) in enumerate(zip(titles, results)):
        plt.subplot(1, 3, i + 1)
        false_positives, miss_rate = result
        plt.plot(false_positives, miss_rate, label=title)
        plt.xscale('log')
        plt.xlabel('false positives per image')
        plt.ylabel('miss rate')
        plt.title(title)
        plt.legend()

    plt.tight_layout()
    plt.savefig(args.evalFig)
    plt.show()
