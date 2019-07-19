echo 'Running'

. ./path.sh

python ./cifar.py \
       --exp exp/cifar/wrn-28-10-ngd \
       -a wrn \
       --depth 28 \
       --widen-factor 10 \
       --optimizer ngd \
       --epochs 50 \
       --scheduler step \
       --milestones 38 \
       --gamma 0.1 \
       --wd 1e-4
