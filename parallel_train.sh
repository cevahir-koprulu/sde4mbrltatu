# Script for performing parallel training of the model on GPU
task = "halfcheetah-medium-expert-v2"
model_name = "hc_me_final2"
rpc = 0.001
seed = 32
rl = 5
cuda_dev = 2
epoch = 2000
mem_frac = 0.1

for cvar in 0.9 0.95 0.99
do
    outdir="tatu_mopo_sde_${task}_rpc_${rpc}_seed_${seed}_rl_${rl}_cvar_${cvar}_diff_density"
    CUDA_VISIBLE_DEVICES=$cuda_dev XLA_PYTHON_CLIENT_MEM_FRACTION=$mem_frac nohup python train_tatu_modelbased.py --task $task --jax_gpu_mem_frac $mem_frac --algo-name tatu_mopo_sde --rollout-length $rl --seed $seed --reward-penalty-coef $rpc --unc_cvar_coef $cvar --threshold_decision_var diff_density --sde_num_particles 1 --use_diffusion --epoch $epoch --model $model_name > $outdir.nohup &
done

# Kill all the processes
# pkill -f train_tatu_modelbased.py