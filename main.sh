models_B=('3' '1.5' '0.5')
checkpoints=('1500' '1000' '500')
one_shots=(0 1)

for model_B in "${models_B[@]}"
do
    CUDA_VISIBLE_DEVICES=5 python test.py --model_B $model_B --checkpoint '' --one_shot 0
    for checkpoint in "${checkpoints[@]}"
    do
        for one_shot in "${one_shots[@]}"
        do
            CUDA_VISIBLE_DEVICES=5 python test.py --model_B $model_B --checkpoint $checkpoint --one_shot $one_shot
        done
    done
done