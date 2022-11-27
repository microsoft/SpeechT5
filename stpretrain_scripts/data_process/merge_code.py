import sys
import torch


def main():
    for line in sys.stdin:
        line = line.rstrip()
        codes = list(map(int, line.split()))
        merged_codes = torch.unique_consecutive(torch.tensor(codes)).numpy()
        merged_codes = map(str, merged_codes)
        print(" ".join(merged_codes))

if __name__ == "__main__":
    main()
