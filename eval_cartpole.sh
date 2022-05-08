# generate a few runs for the cartpole population size evaluation

mrate=0.07
for p_size in 10 50 100 200 500
do
    e1=$(( $p_size / 8))
    e2=$(( $e1 >= 4))

    if [ $e2 = 1 ]; then
        np=4
    else
        np=1
    fi
    python src/main.py --task=cartpole --population_size=$p_size --np=$np --max_generations=50 \
        --mutation_rate=$mrate --save_dir=./carpole_eval/psize:${p_size}_mrate:0.07
done

