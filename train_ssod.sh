#Training phase
for i in 1 2 3 4 5
do
    echo "========= RUNNING TIME: $i ========="
    python train_ssod.py --config config/polyps_set_semi.yaml
done

#Eval phase
for i in 0 1 2 3 4
do 
    echo "========= EVALUATE TIME: $i ========="
    if [ $i -eq 0 ]
    then
        python evaluate.py --config config/polyps_set_semi.yaml --weights save_dir/semi_0.10_20230619_folds/last.ckpt
    else
        python evaluate.py --config config/polyps_set_semi.yaml --weights save_dir/semi_0.10_20230619_folds/last-v$i.ckpt
    fi
done
