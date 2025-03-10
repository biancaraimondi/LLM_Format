models_B=('0.5' '1.5' '3')
checkpoints=('500' '1000' '1500')
one_shots=(true false)

for model_B in "${models_B[@]}"
do
    for checkpoint in "${checkpoints[@]}"
    do
        for one_shot in "${one_shots[@]}"
        do
            python test.py --model_B $model_B --checkpoint $checkpoint --one_shot $one_shot
        done
    done
done