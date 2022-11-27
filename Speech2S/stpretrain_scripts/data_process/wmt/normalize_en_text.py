import re
import sys
import regex
import argparse
from tqdm import tqdm
from num2words import num2words

def writefile(filename, lines):
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, type=str)
    parser.add_argument("--output", "-o", required=True, type=str)
    args = parser.parse_args()
    outlines = []

    with open(f"{args.input}", 'r') as f:
        inputs = f.readlines()

        for line in tqdm(inputs):
            line = line.strip().upper()
            line = re.sub(u"([^\u0041-\u005a\u0061-\u007a\u0030-\u0039\'])", " ", line)
            items = []
            for item in line.split():
                if item.isdigit():
                    try:
                        item = num2words(item)
                    except Exception as e:
                        print(line)
                        raise(e)
                items.append(item)
            line = " ".join(items)
            line = line.replace("-", " ")
            line = line.upper()
            line = line.replace("' S", "'S")
            line = line.replace(" ", "|")
            line = " ".join(line) + " |"
            outlines.append(line + '\n')
            # print(line)

    writefile(args.output, outlines)

if __name__ == "__main__":
    main()
