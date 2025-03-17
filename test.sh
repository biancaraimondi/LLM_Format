#models_B=('7' '3' '1.5' '0.5')
models_B=('7')
checkpoints=('1500' '1000' '500')
one_shots=(0 1 5)

for model_B in "${models_B[@]}"
do
    for one_shot in "${one_shots[@]}"
    CUDA_VISIBLE_DEVICES=5 python test.py --model_B $model_B --checkpoint '' --one_shot $one_shot
    do
        for checkpoint in "${checkpoints[@]}"
        do
            CUDA_VISIBLE_DEVICES=5 python test.py --model_B $model_B --checkpoint $checkpoint --one_shot $one_shot
        done
    done
done