[ $# -lt 2 ] && echo "Usage: $0 <input-text> <outdir> <MODEL> <suffix>" && exit 0

if [ ! -d ${HOME}/sentencepiece ]; then
    CURRENT_DIR=`pwd`
    cd ${HOME}
    git clone https://github.com/google/sentencepiece.git
    cd sentencepiece
    mkdir build && cd build
    cmake .. && make -j 16
    sudo make install
    sudo ldconfig -v
    cd ${HOME}
    cd ${CURRENT_DIR}
fi

input=$1
outdir=$2
MODEL=$3
suffix=$4
outname=${input##*/}
outname=${outname%.wrd*}
[ -z $input ] && echo "You must specify a source file" && exit 1

[ -z $MODEL ] && MODEL=/mnt/default/v-ziqzhang/data/stbert/data/librispeech/hubert_release_iter2_layer9_kmeans/spm_unigram_10000.model && echo "No spm model was specified!, set default to $MODEL"
[ -z $outdir ] && outdir=${input%/*}
[ -z $outdir ] && outdir="."
[ ! -d $outdir ] && mkdir -p $outdir

echo "Output: $outdir/$outname.spm"

echo "------------------------------- tokenize text...--------------------------------------------"
spm_encode --model=$MODEL < ${input} > $outdir/$outname.spm || exit 1
echo "-----------------------------------   done      --------------------------------------------"
