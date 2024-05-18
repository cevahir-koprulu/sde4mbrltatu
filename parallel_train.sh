# Script for performing parallel training of the model on GPU
task="Walker2D-v3-High-1000"
model_name="wk_h_final"
rpc=1.0
cvar=0.98
rl=10
cuda_dev=2
epoch=1000
mem_frac=0.1

for seed in 32 62 92 122 152
do
    outdir="tatu_mopo_sde_${task}_rpc_${rpc}_seed_${seed}_rl_${rl}_cvar_${cvar}_diff_density"
    CUDA_VISIBLE_DEVICES=$cuda_dev XLA_PYTHON_CLIENT_MEM_FRACTION=$mem_frac nohup python train_tatu_modelbased.py --task $task --jax_gpu_mem_frac $mem_frac --algo-name tatu_mopo_sde --seed $seed --reward-penalty-coefficient $rpc --unc_cvar_coeff $cvar --rollout-length $rl --use_diffusion --sde_num_particles --epoch $epoch --model $model_name > $outdir.log &
    sleep 10
done


# for cvar in 0.9 0.95 0.99cvar
# do
#     outdir="tatu_mopo_sde_${task}_rpc_${rpc}_seed_${seed}_rl_${rl}_cvar_${cvar}_diff_density"
#     CUDA_VISIBLE_DEVICES=$cuda_dev XLA_PYTHON_CLIENT_MEM_FRACTION=$mem_frac nohup python train_tatu_modelbased.py --task $task --jax_gpu_mem_frac $mem_frac --algo-name tatu_mopo_sde>
#     sleep 10
# done

# Kill all the processes
# pkill -f train_tatu_modelbased.py