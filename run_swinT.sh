
for seed in 0
do
    echo "Running with random_seed=$seed"
    python MIL_main.py --run_mode test --random_seed ${seed} --batch_size 2 --class_num 3 --bag_weight --bags_len 1042 --num_workers 16\
             --test_weights_feature /data/MIL/MMvC/Weights_Result/Larynx/SwinT_sota_Feature3_ValAcc_0.9057971014492754_Epoch74.pth\
            --test_weights_head /data/MIL/MMvC/Weights_Result/Larynx/SwinT_sota_Head3_ValAcc_0.9057971014492754_Epoch74.pth
done
