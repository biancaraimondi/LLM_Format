#models_B=('7' '3' '1.5' '0.5')
#one_shots=(0 1 5)
models_B=('0.5')
one_shots=(0)

for model_B in "${models_B[@]}"
do
    for one_shot in "${one_shots[@]}"
    do
        CUDA_VISIBLE_DEVICES=5 python train_edit.py --model_B $model_B --one_shot $one_shot
    done
done