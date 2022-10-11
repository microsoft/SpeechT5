import tqdm
import argparse
import numpy as np

sil_prob = 0.25


parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", required=True, type=str)
parser.add_argument("--output", "-o", required=True, type=str)
parser.add_argument("--lexicon", default='/stdblob/users/v-sanych/data/librispeech/librispeech_lm_norm_phn2char/kaldi/lang_nosp/phones/align_lexicon.txt', type=str)
args = parser.parse_args()

dict = {}
f = open(args.lexicon)
for l in f:
    dict[l.split()[0]] = l.split()[2:]
    assert l.split()[0] == l.split()[1]

f = open(args.input, 'r')
w_f = open(f'{args.output}.kaldi_phn_sil025', 'w')
w_oov = open(f'{args.output}.kaldi_phn_sil025.oov', 'w')

oov_nums = 0
total_nums = 0
for l in tqdm.tqdm(f):
    words = l.strip().replace(" ", "").split("|")
    # words = l.strip().upper().split()
    words = [w for w in words if w != '']

    phones = []
    phones.extend(dict['!SIL'])

    sample_sil_probs = None
    if sil_prob > 0 and len(words) > 1:
        sample_sil_probs = np.random.random(len(words) - 1)

    for i, w in enumerate(words):
        total_nums += 1
        if w not in dict:
            w = '<UNK>'
            oov_nums += 1
            w_oov.write(w + '\n')

        phones.extend(dict[w])

        if (
                sample_sil_probs is not None
                and i < len(sample_sil_probs)
                and sample_sil_probs[i] < sil_prob
        ):
            phones.extend(dict['!SIL'])

    phones.extend(dict['!SIL'])
    w_f.write(' '.join(phones) + '\n')
w_oov.write(f'{oov_nums}\n')
print(f"OOV rate: {oov_nums}/{total_nums}")

# !!! After processing, use this comand to adjust the SIL 
### sed -i 's/SIL_S/SIL/g'  your_file
