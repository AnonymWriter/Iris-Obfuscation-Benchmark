device=0

################################## train iris classification model ##################################
# no variation
python train_iris_classification.py -device $device -rp 0 -rd 0 -pp 0 -pd 0

################################## train gaze estimation model ##################################
python train_gaze_estimation.py -device $device -estimator 1 --save_period 10 -E 250
python train_gaze_estimation.py -device $device -estimator 2 --save_period 50 -E 500

##################################  benchmarks on OpenEDS2019 ##################################
python benchmark_openeds2019.py -device $device

##################################  benchmarks on OpenEDS2020 ##################################
python benchmark_openeds2020.py -device $device