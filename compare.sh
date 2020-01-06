cd /home/r07944011/researches/InsightFace_Pytorch
f1=$1
f2=$2
# DIR=$(dirname "${f1}")
# f1_wo_dir=${f1##*/}
# f2_wo_dir=${f2##*/}
# f1_wo_ext=${f1%.*}
# ex_file="${DIR}/${f1_wo_dir}"
python plot_qualitative_results_given_2_imgs.py \
--img1 $1 \
--img2 $2 \
--exdir ./test/ \
--filename $3
