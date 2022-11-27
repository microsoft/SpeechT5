
if [ ! -d ${HOME}/azcopy_linux_amd64_10.11.0 ]; then
    CURRENT_DIR=`pwd`
    cd ${HOME} && wget https://azcopyvnext.azureedge.net/release20210616/azcopy_linux_amd64_10.11.0.tar.gz && tar -zxvf azcopy_linux_amd64_10.11.0.tar.gz && rm -f azcopy_linux_amd64_10.11.0.tar.gz && cd ${CURRENT_DIR}
fi
export PATH=$PATH:${HOME}/azcopy_linux_amd64_10.11.0/:${HOME}/.local/bin
export PYTHONPATH=$PYTHONPATH:/mnt/output/users/v-kunwei/code/fairseq

rank=$1
nshard=$2
split=$3
[ -z $rank ] && echo "please specify rank"
[ -z $nshard ] && nshard=1
[ -z $split ] && split="train"


FAIRSEQ_ROOT=/mnt/output/users/v-kunwei/code/fairseq
ckpt_path=/mnt/output/users/v-kunwei/code/fairseq/examples/speech_to_speech/mhubert_base_vp_en_es_fr_it3.pt
tsv_dir=/home/v-kunwei

feat_dir=${HOME}/$split
python $FAIRSEQ_ROOT/examples/hubert/simple_kmeans/dump_hubert_feature.py ${tsv_dir} ${split} ${ckpt_path} 9 ${nshard} ${rank} ${feat_dir} || exit 1


echo "-------------------------------------------------------------------------------------------"
echo "----------------------------------    done    ---------------------------------------------"
echo "-------------------------------------------------------------------------------------------"

km_path=/mnt/output/users/v-kunwei/code/fairseq/examples/speech_to_speech/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin 
lab_dir=${HOME}/${split}
python $FAIRSEQ_ROOT/examples/hubert/simple_kmeans/dump_km_label.py ${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}


# sas="?sv=2020-08-04&st=2022-01-02T04%3A58%3A15Z&se=2022-06-01T04%3A58%3A00Z&sr=c&sp=racwdl&sig=NyZKOHivgesEoZ8yvLsVT6aZMYQZMevLLmXNOTaWyvU%3D"
# blob="https://msranlcmtteamdrive.blob.core.windows.net/teamdrive/v-ziqzhang/data/stbert/data/librispeech/libri_960/hubert_release_iter2_layer9_kmeans/${split}"
# azcopy copy $feat_dir/${split}_${rank}_${nshard}.len "$blob/$sas"
# azcopy copy $feat_dir/${split}_${rank}_${nshard}.npy "$blob/$sas"
# azcopy copy $lab_dir "$blob/$sas" --recursive



