[ $# -lt 3 ] && echo "Usage: $0 <input-text> <outdir> <DICT> <suffix>" && exit 0

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
DICT=$3
suffix=$4
outname=${input##*/}
outname=${outname%.txt*}
[ -z $input ] && echo "You must specify a source file" && exit 1

[ -z $DICT ] && echo "No dict was specified!" && exit 1
[ -z $outdir ] && outdir=${input%/*}
[ -z $outdir ] && outdir="."
[ ! -d $outdir ] && mkdir -p $outdir

echo "Dict  : $DICT"
echo "------------------------------- creating idx/bin--------------------------------------------"
echo "$input --> $outdir/${outname}${suffix}.idx"
fairseq-preprocess \
  --only-source \
  --trainpref $input \
  --destdir $outdir \
  --thresholdsrc 0 \
  --srcdict ${DICT} \
  --workers 40

mv $outdir/train.idx $outdir/${outname}${suffix}.idx
mv $outdir/train.bin $outdir/${outname}${suffix}.bin
echo "-----------------------------------   done      --------------------------------------------"

