# Neural Stochastic Differential Equations for Uncertainty-Aware Offline RL

Codebase for ICLR 2025 paper [_Neural Stochastic Differential Equations for Uncertainty-Aware Offline RL_](https://openreview.net/forum?id=hxUMQ4fic3&referrer=%5Bthe%20profile%20of%20Cevahir%20Koprulu%5D(%2Fprofile%3Fid%3D~Cevahir_Koprulu1)) by Cevahir Koprulu, Franck Djeumou, and Ufuk Topcu.

Our codebase is built on the repository of _Uncertainty-driven Trajectory Truncation for Data Augmentation in Offline Reinforcement Learning_ (TATU) by Zhang et al. (2023).

Web sources for TATU:

Source code: https://github.com/pipixiaqishi1/TATU

Paper: https://arxiv.org/abs/2304.04660

# Dependencies

- Python 3.9.17
- MuJoCo 2.3.7
- Gym 0.23.1
- D4RL 1.1
- NeoRL 0.3.1
- PyTorch 2.0.1
- TensorFlow 2.14.0
- Jax 0.4.14

# Usage

python train_tatu_modelbased.py --task "hopper-random-v2" --algo-name "tatu_mopo_sde" --reward-penalty-coef 0.001 --rollout-length 10 --unc_cvar_coef 1.0 --seed 32  

For different D4RL & NeoRL tasks, some hyperparametes may be different. Please see the original paper for the list of hyperparameters.

