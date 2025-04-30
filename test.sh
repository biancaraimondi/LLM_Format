#models_B=('7' '3' '1.5' '0.5')
# one_shots=(0 1 5)
# lengths=('_' '_length' '_length_plus')
# datasets=('rosetta' '')
models_B=('7')
checkpoints=('1500' '1000' '500')
one_shots=(0)
lengths=('_length_plus')
datasets=('')

for model_B in "${models_B[@]}"
do
    for one_shot in "${one_shots[@]}"
    do
        for dataset in "${datasets[@]}"
        do
            # CUDA_VISIBLE_DEVICES=5 python test.py --model_B $model_B --checkpoint '' --one_shot $one_shot --length '' --dataset $dataset
            for checkpoint in "${checkpoints[@]}"
            do
                for length in "${lengths[@]}"
                do
                    CUDA_VISIBLE_DEVICES=5 python test.py --model_B $model_B --checkpoint $checkpoint --one_shot $one_shot --length $length --dataset $dataset
                done
            done
        done
    done
done