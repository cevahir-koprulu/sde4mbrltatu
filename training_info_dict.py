TRAINING_INFO_DICT = {
    ########### NeoRL - HalfCheetah ############
    "HalfCheetah-v3-Low-1000-neorl": {
        "tatu_mopo_sde_rew": {
            "model": {
                "name": "hc_l_final_rew",
                "step": -2,
            },
            "config": {
                "RL": 10,
                "CVaR": 1,
                "RPC": 1,
            },
            "seeds": {
                32: "hc_l_final_rew_diff=True_cvar=1.0_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0929_102253",
                62: "hc_l_final_rew_diff=True_cvar=1.0_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0929_102332",
                92: "hc_l_final_rew_diff=True_cvar=1.0_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0929_102337",
                122: "hc_l_final_rew_diff=True_cvar=1.0_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0929_102340",
            },
        },
        "tatu_mopo_sde": {
            "model": {
                "name": "hc_l_final",
                "step": -2,
            },
            "config": {
                "RL": 10,
                "CVaR": 1,
                "RPC": 1,
            },
            "seeds": {
                32: "hc_l_final_diff=True_cvar=1.0_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0517_225453",
                62: "hc_l_final_diff=True_cvar=1.0_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0517_225314",
                92: "hc_l_final_diff=True_cvar=1.0_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0517_225317",
                122: "hc_l_final_diff=True_cvar=1.0_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0517_225316",
            },
        },
    },
    "HalfCheetah-v3-Medium-1000-neorl": {
        "tatu_mopo_sde_rew": {
            "model": {
                "name": "hc_m_final_rew",
                "step": -2,
            },
            "config": {
                "RL": 10,
                "CVaR": 0.99,
                "RPC": 1,
            },
            "seeds": {
                32: "hc_m_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0929_103044",
                62: "hc_m_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0929_103051",
                92: "hc_m_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0929_103051",
                122: "hc_m_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0929_103046",
            },
        },
        "tatu_mopo_sde": {
            "model": {
                "name": "hc_m_final",
                "step": -2,
            },
            "config": {
                "RL": 10,
                "CVaR": 0.99,
                "RPC": 1,
            },
            "seeds": {
                32: "hc_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0517_014642",
                62: "hc_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0517_014647",
                92: "hc_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0517_014659",
                122: "hc_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0517_014707",
            },
        },
    },
    "HalfCheetah-v3-High-1000-neorl": {
        "tatu_mopo_sde_rew": {
            "model": {
                "name": "hc_h_final_rew",
                "step": -2,
            },
            "config": {
                "RL": 5,
                "CVaR": 0.9,
                "RPC": 0.1,
            },
            "seeds": {
                32: "hc_h_final_rew_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=0.1_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0930_011825",
                62: "hc_h_final_rew_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=0.1_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0930_011906",
                92: "hc_h_final_rew_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=0.1_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0930_011912",
                122: "hc_h_final_rew_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=0.1_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0930_011914",
            },
        },
        "tatu_mopo_sde": {
            "model": {
                "name": "hc_h_final",
                "step": -2,
            },
            "config": {
                "RL": 5,
                "CVaR": 0.9,
                "RPC": 0.1,
            },
            "seeds": {
                32: "hc_h_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=0.1_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0517_225516",
                62: "hc_h_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=0.1_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0517_225539",
                92: "hc_h_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=0.1_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0517_225523",
                122: "hc_h_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=0.1_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0517_225521",
            },
        },
    },
    ########### NeoRL - Hopper ############
    "Hopper-v3-Low-1000-neorl": {
        "tatu_mopo_sde_rew": {
            "model": {
                "name": "hop_l_final_rew",
                "step": -2,
            },
            "config": {
                "RL": 10,
                "CVaR": 0.99,
                "RPC": 0.001,
            },
            "seeds": {
                32: "hop_l_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0928_231632",
                62: "hop_l_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0928_231636",
                92: "hop_l_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0928_231640",
                122: "hop_l_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0928_231644",
            },
        },
        "tatu_mopo_sde": {
            "model": {
                "name": "hop_l_final",
                "step": -2,
            },
            "config": {
                "RL": 10,
                "CVaR": 0.99,
                "RPC": 0.001,
            },
            "seeds": {
                32: "hop_l_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0517_020708",
                62: "hop_l_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0517_020723",
                92: "hop_l_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0517_020728",
                122: "hop_l_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0517_131938",
            },
        },
    },
    "Hopper-v3-Medium-1000-neorl": {
        "tatu_mopo_sde_rew": {
            "model": {
                "name": "hop_m_final_rew",
                "step": -2,
            },
            "config": {
                "RL": 5,
                "CVaR": 0.9,
                "RPC": 0.1,
            },
            "seeds": {
                32: "hop_m_final_rew_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=0.1_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0929_102543",
                62: "hop_m_final_rew_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=0.1_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0929_102547",
                92: "hop_m_final_rew_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=0.1_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0929_102549",
                122: "hop_m_final_rew_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=0.1_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0929_102552",
            },
        },
        "tatu_mopo_sde": {
            "model": {
                "name": "hop_m_final",
                "step": -2,
            },
            "config": {
                "RL": 5,
                "CVaR": 0.9,
                "RPC": 0.1,
            },
            "seeds": {
                32: "hop_m_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=0.1_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0517_131748",
                62: "hop_m_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=0.1_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0517_131758",
                92: "hop_m_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=0.1_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0517_131800",
                122: "hop_m_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=0.1_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0517_131812",
            },
        },
    },
    "Hopper-v3-High-1000-neorl": {
        "tatu_mopo_sde_rew": {
            "model": {
                "name": "hop_h_final_rew",
                "step": -2,
            },
            "config": {
                "RL": 5,
                "CVaR": 0.9,
                "RPC": 0.1,
            },
            "seeds": {
                32: "hop_h_final_rew_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=0.1_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0930_011958",
                62: "hop_h_final_rew_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=0.1_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0930_012005",
                92: "hop_h_final_rew_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=0.1_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0930_012009",
                122: "hop_h_final_rew_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=0.1_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0930_012013",
            },
        },
        "tatu_mopo_sde": {
            "model": {
                "name": "hop_h_final",
                "step": -2,
            },
            "config": {
                "RL": 5,
                "CVaR": 0.9,
                "RPC": 0.1,
            },
            "seeds": {
                32: "hop_h_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=0.1_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0517_224049",
                62: "hop_h_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=0.1_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0517_224054",
                92: "hop_h_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=0.1_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0517_224056",
                122: "hop_h_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=0.1_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0517_224059",
            },
        },
    },
    ########### NeoRL - Walker2d ############
    "Walker2d-v3-Low-1000-neorl": {
        "tatu_mopo_sde_rew": {
            "model": {
                "name": "wk_l_final_rew",
                "step": -2,
            },
            "config": {
                "RL": 5,
                "CVaR": 0.99,
                "RPC": 0.001,
            },
            "seeds": {
                32: "wk_l_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0928_231500",
                62: "wk_l_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0928_231505",
                92: "wk_l_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0928_231515",
                122: "wk_l_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0928_231520",
            },
        },
        "tatu_mopo_sde": {
            "model": {
                "name": "wk_l_final",
                "step": -2,
            },
            "config": {
                "RL": 5,
                "CVaR": 0.99,
                "RPC": 0.001,
            },
            "seeds": {
                32: "wk_l_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0517_015600",
                62: "wk_l_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0517_015637",
                92: "wk_l_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0517_015643",
                122: "wk_l_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0517_015652",
            },
        },
    },
    "Walker2d-v3-Medium-1000-neorl": {
        "tatu_mopo_sde_rew": {
            "model": {
                "name": "wk_m_final_rew",
                "step": -2,
            },
            "config": {
                "RL": 5,
                "CVaR": 0.99,
                "RPC": 0.001,
            },
            "seeds": {
                32: "wk_m_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0929_102756",
                62: "wk_m_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0929_102759",
                92: "wk_m_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0929_102803",
                122: "wk_m_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0929_102800",
            },
        },
        "tatu_mopo_sde": {
            "model": {
                "name": "wk_m_final",
                "step": -2,
            },
            "config": {
                "RL": 5,
                "CVaR": 0.99,
                "RPC": 0.001,
            },
            "seeds": {
                32: "wk_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0517_021224",
                62: "wk_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0517_021322",
                92: "wk_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0517_021328",
                122: "wk_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0517_021335",
            },
        },
    },
    "Walker2d-v3-High-1000-neorl": {
        "tatu_mopo_sde_rew": {
            "model": {
                "name": "wk_h_final_rew",
                "step": -2,
            },
            "config": {
                "RL": 5,
                "CVaR": 0.99,
                "RPC": 1,
            },
            "seeds": {
                32: "wk_h_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0930_012140",
                62: "wk_h_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0930_012146",
                92: "wk_h_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0930_012207",
                122: "wk_h_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0930_012211",
            },
        },
        "tatu_mopo_sde": {
            "model": {
                "name": "wk_h_final",
                "step": -2,
            },
            "config": {
                "RL": 5,
                "CVaR": 0.99,
                "RPC": 1,
            },
            "seeds": {
                32: "wk_h_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0517_020135",
                62: "wk_h_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0517_020137",
                92: "wk_h_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0517_020139",
                122: "wk_h_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0517_020142",
            },
        },
    },
    ##### D4RL - HalfCheetah #####
    "halfcheetah-random-v2": {
        "tatu_mopo_sde_disc": {  
            "model": {
                "name": "hc_rand_final",
                "step": -2,
            },
            "config": {
                "RL": 20,
                "CVaR": 1,
                "RPC": 0.001,
            },
            "seeds": {
                32: "hc_rand_final_diff=True_cvar=1.0_tdv=disc_rl=20_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_1116_191649",
                62: "hc_rand_final_diff=True_cvar=1.0_tdv=disc_rl=20_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_1116_191702",
                92: "hc_rand_final_diff=True_cvar=1.0_tdv=disc_rl=20_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_1120_164927",
                122: "hc_rand_final_diff=True_cvar=1.0_tdv=disc_rl=20_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_1120_164935",
            },
        },
        "tatu_mopo_sde_dfd": {  
            "model": {
                "name": "hc_rand_final",
                "step": -2,
            },
            "config": {
                "RL": 20,
                "CVaR": 1,
                "RPC": 0.001,
            },
            "seeds": {
                32: "hc_rand_final_diff=True_cvar=1.0_tdv=dad_free_diff_rl=20_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_1116_184412",
                62: "hc_rand_final_diff=True_cvar=1.0_tdv=dad_free_diff_rl=20_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_1116_184418",
                92: "hc_rand_final_diff=True_cvar=1.0_tdv=diff_density_rl=20_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0914_143029",
                122: "hc_rand_final_diff=True_cvar=1.0_tdv=dad_free_diff_rl=20_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_1120_164912",
            },
        },
        "tatu_mopo_sde_rew": {  
            "model": {
                "name": "hc_rand_final_rew",
                "step": -2,
            },
            "config": {
                "RL": 20,
                "CVaR": 1,
                "RPC": 0.001,
            },
            "seeds": {
                32: "hc_rand_final_rew_diff=True_cvar=1.0_tdv=diff_density_rl=20_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0922_132212",
                62: "hc_rand_final_rew_diff=True_cvar=1.0_tdv=diff_density_rl=20_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0922_132215",
                92: "hc_rand_final_rew_diff=True_cvar=1.0_tdv=diff_density_rl=20_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0922_132220",
                122: "hc_rand_final_rew_diff=True_cvar=1.0_tdv=diff_density_rl=20_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0922_132222",
                152: "hc_rand_final_rew_diff=True_cvar=1.0_tdv=diff_density_rl=20_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0922_132226",
            },
        },
        "tatu_mopo_sde": {  
            "model": {
                "name": "hc_rand_final",
                "step": -2,
            },
            "config": {
                "RL": 20,
                "CVaR": 1,
                "RPC": 0.001,
            },
            "seeds": {
                32: "hc_rand_final_diff=True_cvar=1.0_tdv=diff_density_rl=20_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0914_143008",
                62: "hc_rand_final_diff=True_cvar=1.0_tdv=diff_density_rl=20_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0914_143021",
                92: "hc_rand_final_diff=True_cvar=1.0_tdv=diff_density_rl=20_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0914_143029",
                122: "hc_rand_final_diff=True_cvar=1.0_tdv=diff_density_rl=20_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0914_143037",
                152: "hc_rand_final_diff=True_cvar=1.0_tdv=diff_density_rl=20_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0914_143042",
            },
        },
        "tatu_mopo":{
            "model": {
                "name": "critic_num_2_seed_32_0518_155708-halfcheetah_random_v2_tatu_mopo",
            },
            "config": {
                "RL": 5,
                "PC": 2,
                "RPC": 1,
            },
            "seeds": {
                32: "critic_num_2_seed_32_0518_155708-halfcheetah_random_v2_tatu_mopo",
                62: "critic_num_2_seed_62_0518_155712-halfcheetah_random_v2_tatu_mopo",
                92: "critic_num_2_seed_92_0518_155715-halfcheetah_random_v2_tatu_mopo",
                122: "critic_num_2_seed_122_0518_155716-halfcheetah_random_v2_tatu_mopo",
                152: "critic_num_2_seed_152_0518_155719-halfcheetah_random_v2_tatu_mopo",
            },
        },
        "mopo":{
            "model": {
                "name": "critic_num_2_seed_32_0519_152820_pc=1e-06-halfcheetah_random_v2_tatu_mopo",
            },
            "config": {
                "RL": 5,
                "RPC": 0.5,
            },
            "seeds": {
                32: "critic_num_2_seed_32_0519_152820_pc=1e-06-halfcheetah_random_v2_tatu_mopo",
                62: "critic_num_2_seed_62_0519_152840_pc=1e-06-halfcheetah_random_v2_tatu_mopo",
                92: "critic_num_2_seed_92_0519_152843_pc=1e-06-halfcheetah_random_v2_tatu_mopo",
                122: "critic_num_2_seed_122_0519_152858_pc=1e-06-halfcheetah_random_v2_tatu_mopo",
                152: "critic_num_2_seed_152_0519_152907_pc=1e-06-halfcheetah_random_v2_tatu_mopo",
            },
        },
    },
    "halfcheetah-medium-v2": {
        "tatu_mopo_sde_rew": {
            "model": {
                "name": "hc_m_final_rew",
                "step": -2,
            },
            "config": {
                "RL": 5,
                "CVaR": 0.99,
                "RPC": 1,
            },
            "seeds": {
                32: "hc_m_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0922_132341",
                62: "hc_m_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0922_132350",
                92: "hc_m_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0922_132422",
                122: "hc_m_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0922_132426",
                152: "hc_m_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0922_132429",
            },
        },
        "tatu_mopo_sde": {
            "model": {
                "name": "hc_m_final",
                "step": -2,
            },
            "config": {
                "RL": 5,
                "CVaR": 0.99,
                "RPC": 1,
            },
            "seeds": {
                32: "hc_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0914_143250",
                62: "hc_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0914_143335",
                92: "hc_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0914_143415",
                122: "hc_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0914_143429",
                152: "hc_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0914_143452",
            },
        },
        "tatu_mopo":{
            "model": {
                "name": "critic_num_2_seed_32_0518_155450-halfcheetah_medium_v2_tatu_mopo",
            },
            "config": {
                "RL": 5,
                "PC": 2,
                "RPC": 1,
            },
            "seeds": {
                32: "critic_num_2_seed_32_0518_155450-halfcheetah_medium_v2_tatu_mopo",
                62: "critic_num_2_seed_62_0518_155501-halfcheetah_medium_v2_tatu_mopo",
                92: "critic_num_2_seed_92_0518_155509-halfcheetah_medium_v2_tatu_mopo",
                122: "critic_num_2_seed_122_0518_155514-halfcheetah_medium_v2_tatu_mopo",
                152: "critic_num_2_seed_152_0518_155524-halfcheetah_medium_v2_tatu_mopo",
            },
        },
        "mopo":{
            "model": {
                "name": "critic_num_2_seed_32_0519_153029_pc=1e-06-halfcheetah_medium_v2_tatu_mopo",
            },
            "config": {
                "RL": 1,
                "RPC": 1,
            },
            "seeds": {
                32: "critic_num_2_seed_32_0519_153029_pc=1e-06-halfcheetah_medium_v2_tatu_mopo",
                62: "critic_num_2_seed_62_0519_153032_pc=1e-06-halfcheetah_medium_v2_tatu_mopo",
                92: "critic_num_2_seed_92_0519_153040_pc=1e-06-halfcheetah_medium_v2_tatu_mopo",
                122: "critic_num_2_seed_122_0519_153053_pc=1e-06-halfcheetah_medium_v2_tatu_mopo",
                152: "critic_num_2_seed_152_0519_153056_pc=1e-06-halfcheetah_medium_v2_tatu_mopo",
            },
        },
    },
    "halfcheetah-medium-replay-v2": {
        "tatu_mopo_sde_rew": {
            "model": {
                "name": "hc_mr_final_rew",
                "step": -2,
            },
            "config": {
                "RL": 5,
                "CVaR": 0.9,
                "RPC": 1,
            },
            "seeds": {
                32: "hc_mr_final_rew_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0922_132512",
                62: "hc_mr_final_rew_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0922_132522",
                92: "hc_mr_final_rew_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0922_132533",
                122: "hc_mr_final_rew_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0922_132538",
                152: "hc_mr_final_rew_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0922_132543",

            },
        },
        "tatu_mopo_sde": {
            "model": {
                "name": "hc_mr_final",
                "step": -2,
            },
            "config": {
                "RL": 5,
                "CVaR": 0.9,
                "RPC": 1,
            },
            "seeds": {
                32: "hc_mr_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0914_143725",
                62: "hc_mr_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0914_143740",
                92: "hc_mr_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0914_143747",
                122: "hc_mr_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0914_143753",
                152: "hc_mr_final_diff=True_cvar=0.9_tdv=diff_density_rl=5_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0914_143758",

            },
        },
        "tatu_mopo":{
            "model": {
                "name": "critic_num_2_seed_32_0518_155620-halfcheetah_medium_replay_v2_tatu_mopo",
            },
            "config": {
                "RL": 5,
                "PC": 2,
                "RPC": 1,
            },
            "seeds": {
                32: "critic_num_2_seed_32_0518_155620-halfcheetah_medium_replay_v2_tatu_mopo",
                62:  "critic_num_2_seed_62_0518_155545-halfcheetah_medium_replay_v2_tatu_mopo",
                92: "critic_num_2_seed_92_0518_155548-halfcheetah_medium_replay_v2_tatu_mopo",
                122: "critic_num_2_seed_122_0518_155550-halfcheetah_medium_replay_v2_tatu_mopo",
                152: "critic_num_2_seed_152_0518_155555-halfcheetah_medium_replay_v2_tatu_mopo",
            },
        },
        "mopo":{
            "model": {
                "name": "critic_num_2_seed_32_0519_153143_pc=1e-06-halfcheetah_medium_replay_v2_tatu_mopo",
            },
            "config": {
                "RL": 5,
                "RPC": 1,
            },
            "seeds": {
                32: "critic_num_2_seed_32_0519_153143_pc=1e-06-halfcheetah_medium_replay_v2_tatu_mopo",
                62: "critic_num_2_seed_62_0519_153149_pc=1e-06-halfcheetah_medium_replay_v2_tatu_mopo",
                92: "critic_num_2_seed_92_0519_153212_pc=1e-06-halfcheetah_medium_replay_v2_tatu_mopo",
                122: "critic_num_2_seed_122_0519_153216_pc=1e-06-halfcheetah_medium_replay_v2_tatu_mopo",
                152:  "critic_num_2_seed_152_0519_153224_pc=1e-06-halfcheetah_medium_replay_v2_tatu_mopo",
            },
        },
    },
    "halfcheetah-medium-expert-v2": {
        "tatu_mopo_sde_disc": {
            "model": {
                "name": "hc_me_final",
                "step": -2,
            },
            "config": {
                "RL": 10,
                "CVaR": 0.95,
                "RPC": 1,
            },
            "seeds": {
                32: "hc_me_final_diff=True_cvar=0.95_tdv=disc_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_1116_192450",
                62: "hc_me_final_diff=True_cvar=0.95_tdv=disc_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_1116_192458",
                92: "hc_me_final_diff=True_cvar=0.95_tdv=disc_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_1120_164533",
                122: "hc_me_final_diff=True_cvar=0.95_tdv=disc_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_1120_164546",
            },
        },
        "tatu_mopo_sde_dfd": {
            "model": {
                "name": "hc_me_final",
                "step": -2,
            },
            "config": {
                "RL": 10,
                "CVaR": 0.95,
                "RPC": 1,
            },
            "seeds": {
                32: "hc_me_final_diff=True_cvar=0.95_tdv=dad_free_diff_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_1116_192426",
                62: "hc_me_final_diff=True_cvar=0.95_tdv=dad_free_diff_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_1116_192435",
                92: "hc_me_final_diff=True_cvar=0.95_tdv=dad_free_diff_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_1120_164440",
                122: "hc_me_final_diff=True_cvar=0.95_tdv=dad_free_diff_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_1120_164448",
            },
        },
        "tatu_mopo_sde_rew": {
            "model": {
                "name": "hc_me_final_rew",
                "step": -2,
            },
            "config": {
                "RL": 10,
                "CVaR": 0.95,
                "RPC": 1,
            },
            "seeds": {
                32: "hc_me_final_rew_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0922_132729",
                62: "hc_me_final_rew_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0922_132736",
                92: "hc_me_final_rew_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0922_132742",
                122: "hc_me_final_rew_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0922_132731",
                152: "hc_me_final_rew_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0922_132740",
            },
        },
        "tatu_mopo_sde": {
            "model": {
                "name": "hc_me_final",
                "step": -2,
            },
            "config": {
                "RL": 10,
                "CVaR": 0.95,
                "RPC": 1,
            },
            "seeds": {
                32: "hc_me_final_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0914_143928",
                62: "hc_me_final_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0914_143931",
                92: "hc_me_final_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0914_144004",
                122: "hc_me_final_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0914_143937",
                152: "hc_me_final_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0914_143916",
            },
        },
        "tatu_mopo":{
            "model": {
                "name": "critic_num_2_seed_32_0518_155659-halfcheetah_medium_expert_v2_tatu_mopo",
            },
            "config": {
                "RL": 5,
                "PC": 2,
                "RPC": 1,
            },
            "seeds": {
                32: "critic_num_2_seed_32_0518_155659-halfcheetah_medium_expert_v2_tatu_mopo",
                62: "critic_num_2_seed_62_0518_155702-halfcheetah_medium_expert_v2_tatu_mopo",
                92: "critic_num_2_seed_92_0518_155703-halfcheetah_medium_expert_v2_tatu_mopo",
                122: "critic_num_2_seed_122_0518_155708-halfcheetah_medium_expert_v2_tatu_mopo",
                152: "critic_num_2_seed_152_0518_155712-halfcheetah_medium_expert_v2_tatu_mopo",
            },
        },
        "mopo":{
            "model": {
                "name": "critic_num_2_seed_32_0519_153307_pc=1e-06-halfcheetah_medium_expert_v2_tatu_mopo",
            },
            "config": {
                "RL": 5,
                "RPC": 1,
            },
            "seeds": {
                32: "critic_num_2_seed_32_0519_153307_pc=1e-06-halfcheetah_medium_expert_v2_tatu_mopo",
                62: "critic_num_2_seed_62_0519_153330_pc=1e-06-halfcheetah_medium_expert_v2_tatu_mopo",
                92: "critic_num_2_seed_92_0519_153335_pc=1e-06-halfcheetah_medium_expert_v2_tatu_mopo",
                122: "critic_num_2_seed_122_0519_153339_pc=1e-06-halfcheetah_medium_expert_v2_tatu_mopo",
                152: "critic_num_2_seed_152_0519_153343_pc=1e-06-halfcheetah_medium_expert_v2_tatu_mopo",
            },
        },
    },
   ##### D4RL - Hopper #####
    "hopper-random-v2": {
        "tatu_mopo_sde_disc": {  
            "model": {
                "name": "hop_rand_final",
                "step": -2,
            },
            "config": {
                "RL": 10,
                "CVaR": 1,
                "RPC": 0.001,
            },
            "seeds": {
                32: "hop_rand_final_diff=True_cvar=1.0_tdv=disc_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_1118_233713",
                62: "hop_rand_final_diff=True_cvar=1.0_tdv=disc_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_1118_233723",
                92: "hop_rand_final_diff=True_cvar=1.0_tdv=disc_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_1120_144605",
                122: "hop_rand_final_diff=True_cvar=1.0_tdv=disc_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_1120_144613",
            },
        },
        "tatu_mopo_sde_dfd": {  
            "model": {
                "name": "hop_rand_final",
                "step": -2,
            },
            "config": {
                "RL": 10,
                "CVaR": 1,
                "RPC": 0.001,
            },
            "seeds": {
                32: "hop_rand_final_diff=True_cvar=1.0_tdv=dad_free_diff_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_1118_233655",
                62: "hop_rand_final_diff=True_cvar=1.0_tdv=dad_free_diff_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_1118_233700",
                92: "hop_rand_final_diff=True_cvar=1.0_tdv=dad_free_diff_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_1120_144548",
                122: "hop_rand_final_diff=True_cvar=1.0_tdv=dad_free_diff_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_1120_144555",
            },
        },
        "tatu_mopo_sde_rew": { 
            "model": {
                "name": "hop_rand_final_rew",
                "step": -2,
            }, 
            "config": {
                "RL": 10,
                "CVaR": 1,
                "RPC": 0.001,
            },
            "seeds": {
                32: "hop_rand_final_rew_diff=True_cvar=1.0_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0927_233446",
                62: "hop_rand_final_rew_diff=True_cvar=1.0_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0927_233453",
                92: "hop_rand_final_rew_diff=True_cvar=1.0_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0927_233500",
                122: "hop_rand_final_rew_diff=True_cvar=1.0_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0927_233503",
                152: "hop_rand_final_rew_diff=True_cvar=1.0_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0927_233507",
            },
        },
        "tatu_mopo_sde": {  
            "model": {
                "name": "hop_rand_final",
                "step": -2,
            },
            "config": {
                "RL": 10,
                "CVaR": 1,
                "RPC": 0.001,
            },
            "seeds": {
                32: "hop_rand_final_diff=True_cvar=1.0_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0918_205144",
                62: "hop_rand_final_diff=True_cvar=1.0_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0918_205322",
                92: "hop_rand_final_diff=True_cvar=1.0_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0918_205329",
                122: "hop_rand_final_diff=True_cvar=1.0_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0918_205334",
                152: "hop_rand_final_diff=True_cvar=1.0_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0918_205339",
            },
        },
        "tatu_mopo":{
            "model": {
                "name": "critic_num_2_seed_32_0919_153111_pc=2.0-hopper_random_v2_tatu_mopo",
            },
            "config": {
                "RL": 5,
                "PC": 2,
                "RPC": 1,
            },
            "seeds": {
                32: "critic_num_2_seed_32_0919_153111_pc=2.0-hopper_random_v2_tatu_mopo",
                62: "critic_num_2_seed_62_0919_153232_pc=2.0-hopper_random_v2_tatu_mopo",
                92: "critic_num_2_seed_92_0919_153238_pc=2.0-hopper_random_v2_tatu_mopo",
                122: "critic_num_2_seed_122_0919_153759_pc=2.0-hopper_random_v2_tatu_mopo",
                152: "critic_num_2_seed_152_0920_195904_pc=2.0-hopper_random_v2_tatu_mopo",
            },
        },
        "mopo":{
            "model": {
                "name": "critic_num_2_seed_32_0926_164351_pc=1e-06-hopper_random_v2_tatu_mopo",
            },
            "config": {
                "RL": 5,
                "RPC": 1,
            },
            "seeds": {
                32: "critic_num_2_seed_32_0926_164351_pc=1e-06-hopper_random_v2_tatu_mopo",
                62: "critic_num_2_seed_62_0926_164428_pc=1e-06-hopper_random_v2_tatu_mopo",
                92: "critic_num_2_seed_92_0926_164434_pc=1e-06-hopper_random_v2_tatu_mopo",
                122: "critic_num_2_seed_122_0926_164440_pc=1e-06-hopper_random_v2_tatu_mopo",
                152: "critic_num_2_seed_152_0928_010931_pc=1e-06-hopper_random_v2_tatu_mopo",
            },
        },
    },
    "hopper-medium-v2": {
        "tatu_mopo_sde_rew": {
            "model": {
                "name": "hop_m_final_rew",
                "step": -2,
            },
            "config": {
                "RL": 10,
                "CVaR": 0.99,
                "RPC": 1,
            },
            "seeds": {
                32: "hop_m_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0927_233636",
                62: "hop_m_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0927_233710",
                92: "hop_m_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0927_233727",
                122: "hop_m_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0927_233742",
                152: "hop_m_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0927_233747",
            },
        },
        "tatu_mopo_sde": {
            "model": {
                "name": "hop_m_final",
                "step": -2,
            },
            "config": {
                "RL": 10,
                "CVaR": 0.99,
                "RPC": 1,
            },
            "seeds": {
                32: "hop_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0918_205524",
                62: "hop_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0918_205623",
                92: "hop_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0918_205630",
                122: "hop_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0918_205636",
                152: "hop_m_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0918_205641",
            },
        },
        "tatu_mopo":{
            "model": {
                "name": "critic_num_2_seed_32_0919_153900_pc=3.5-hopper_medium_v2_tatu_mopo",
            },
            "config": {
                "RL": 5,
                "PC": 3.5,
                "RPC": 1,
            },
            "seeds": {
                32: "critic_num_2_seed_32_0919_153900_pc=3.5-hopper_medium_v2_tatu_mopo",
                62: "critic_num_2_seed_62_0919_153907_pc=3.5-hopper_medium_v2_tatu_mopo",
                92: "critic_num_2_seed_92_0919_153912_pc=3.5-hopper_medium_v2_tatu_mopo",
                122: "critic_num_2_seed_122_0919_153916_pc=3.5-hopper_medium_v2_tatu_mopo",
                152: "critic_num_2_seed_152_0920_195925_pc=3.5-hopper_medium_v2_tatu_mopo",
            },
        },
        "mopo":{
            "model": {
                "name": "critic_num_2_seed_32_0928_175605_pc=1e-06-hopper_medium_v2_tatu_mopo",
            },
            "config": {
                "RL": 5,
                "RPC": 5,
            },
            "seeds": {
                32: "critic_num_2_seed_32_0928_175605_pc=1e-06-hopper_medium_v2_tatu_mopo",
                62: "critic_num_2_seed_62_0928_175616_pc=1e-06-hopper_medium_v2_tatu_mopo",
                92: "critic_num_2_seed_92_0928_175621_pc=1e-06-hopper_medium_v2_tatu_mopo",
                122: "critic_num_2_seed_122_0928_175625_pc=1e-06-hopper_medium_v2_tatu_mopo",
                152: "critic_num_2_seed_152_0928_175634_pc=1e-06-hopper_medium_v2_tatu_mopo",
            },
        },
    },
    "hopper-medium-replay-v2": {
        "tatu_mopo_sde_rew": {
            "model": {
                "name": "hop_mr_final_rew",
                "step": -2,
            },
            "config": {
                "RL": 10,
                "CVaR": 0.99,
                "RPC": 1,
            },
            "seeds": {
                32: "hop_mr_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0927_233819",
                62: "hop_mr_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0927_233824",
                92: "hop_mr_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0927_233830",
                122: "hop_mr_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0927_233827",
                152: "hop_mr_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0927_233832",

            },
        },
        "tatu_mopo_sde": {
            "model": {
                "name": "hop_mr_final",
                "step": -2,
            },
            "config": {
                "RL": 10,
                "CVaR": 0.99,
                "RPC": 1,
            },
            "seeds": {
                32: "hop_mr_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0918_205953",
                62: "hop_mr_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0918_205959",
                92: "hop_mr_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0918_210004",
                122: "hop_mr_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0918_210007",
                152: "hop_mr_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0918_210012",

            },
        },
        "tatu_mopo":{
            "model": {
                "name": "critic_num_2_seed_32_0920_195703_pc=2.0-hopper_medium_replay_v2_tatu_mopo",
            },
            "config": {
                "RL": 5,
                "PC": 2,
                "RPC": 1,
            },
            "seeds": {
                32: "critic_num_2_seed_32_0920_195703_pc=2.0-hopper_medium_replay_v2_tatu_mopo",
                62: "critic_num_2_seed_62_0920_195715_pc=2.0-hopper_medium_replay_v2_tatu_mopo",
                92: "critic_num_2_seed_92_0920_195724_pc=2.0-hopper_medium_replay_v2_tatu_mopo",
                122: "critic_num_2_seed_122_0920_195728_pc=2.0-hopper_medium_replay_v2_tatu_mopo",
                152: "critic_num_2_seed_152_0920_195940_pc=2.0-hopper_medium_replay_v2_tatu_mopo",
            },
        },
        "mopo":{
            "model": {
                "name": "critic_num_2_seed_32_0926_164754_pc=1e-06-hopper_medium_replay_v2_tatu_mopo",
            },
            "config": {
                "RL": 5,
                "RPC": 1,
            },
            "seeds": {
                32: "critic_num_2_seed_32_0926_164754_pc=1e-06-hopper_medium_replay_v2_tatu_mopo",
                62: "critic_num_2_seed_62_0926_164800_pc=1e-06-hopper_medium_replay_v2_tatu_mopo",
                92: "critic_num_2_seed_92_0926_165046_pc=1e-06-hopper_medium_replay_v2_tatu_mopo",
                122: "critic_num_2_seed_122_0926_165136_pc=1e-06-hopper_medium_replay_v2_tatu_mopo",
                152: "critic_num_2_seed_152_0926_165151_pc=1e-06-hopper_medium_replay_v2_tatu_mopo",
            },
        },
    },
    "hopper-medium-expert-v2": {
        "tatu_mopo_sde_disc": {
            "model": {
                "name": "hop_me_final",
                "step": -2,
            },
            "config": {
                "RL": 10,
                "CVaR": 0.99,
                "RPC": 1,
            },
            "seeds": {
                32: "hop_me_final_diff=True_cvar=0.99_tdv=disc_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_1118_140314",
                62: "hop_me_final_diff=True_cvar=0.99_tdv=disc_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_1118_140325",
                92: "hop_me_final_diff=True_cvar=0.99_tdv=disc_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_1120_144418",
                122: "hop_me_final_diff=True_cvar=0.99_tdv=disc_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_1120_144420",
            },
        },
        "tatu_mopo_sde_dfd": {
            "model": {
                "name": "hop_me_final",
                "step": -2,
            },
            "config": {
                "RL": 10,
                "CVaR": 0.99,
                "RPC": 1,
            },
            "seeds": {
                32: "hop_me_final_diff=True_cvar=0.99_tdv=dad_free_diff_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_1118_140247",
                62: "hop_me_final_diff=True_cvar=0.99_tdv=dad_free_diff_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_1118_140258",
                92: "hop_me_final_diff=True_cvar=0.99_tdv=dad_free_diff_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_1120_144408",
                122: "hop_me_final_diff=True_cvar=0.99_tdv=dad_free_diff_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_1120_144411",
            },
        },
        "tatu_mopo_sde_rew": {
            "model": {
                "name": "hop_me_final_rew",
                "step": -2,
            },
            "config": {
                "RL": 10,
                "CVaR": 0.99,
                "RPC": 1,
            },
            "seeds": {
                32: "hop_me_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0926_163459",
                62: "hop_me_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0926_163503",
                92: "hop_me_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0926_163510",
                122: "hop_me_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0926_163513",
                152: "hop_me_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0926_163515",
            },
        },
        "tatu_mopo_sde": {
            "model": {
                "name": "hop_me_final",
                "step": -2,
            },
            "config": {
                "RL": 10,
                "CVaR": 0.99,
                "RPC": 1,
            },
            "seeds": {
                32: "hop_me_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0918_210158",
                62: "hop_me_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0918_210310",
                92: "hop_me_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0918_210319",
                122: "hop_me_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0918_210323",
                152: "hop_me_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0918_210330",
            },
        },
        "tatu_mopo":{
            "model": {
                "name": "critic_num_2_seed_32_0920_195817_pc=3.5-hopper_medium_expert_v2_tatu_mopo",
            },
            "config": {
                "RL": 5,
                "PC": 3.5,
                "RPC": 1,
            },
            "seeds": {
                32: "critic_num_2_seed_32_0920_195817_pc=3.5-hopper_medium_expert_v2_tatu_mopo",
                62: "critic_num_2_seed_62_0920_195824_pc=3.5-hopper_medium_expert_v2_tatu_mopo",
                92: "critic_num_2_seed_92_0920_195828_pc=3.5-hopper_medium_expert_v2_tatu_mopo",
                122: "critic_num_2_seed_122_0920_195833_pc=3.5-hopper_medium_expert_v2_tatu_mopo",
                152: "critic_num_2_seed_152_0920_200022_pc=3.5-hopper_medium_expert_v2_tatu_mopo",
            },
        },
        "mopo":{
            "model": {
                "name": "critic_num_2_seed_32_0926_165253_pc=1e-06-hopper_medium_expert_v2_tatu_mopo",
            },
            "config": {
                "RL": 5,
                "RPC": 1,
            },
            "seeds": {
                32: "critic_num_2_seed_32_0926_165253_pc=1e-06-hopper_medium_expert_v2_tatu_mopo",
                62: "critic_num_2_seed_62_0926_165231_pc=1e-06-hopper_medium_expert_v2_tatu_mopo",
                92: "critic_num_2_seed_92_0926_165240_pc=1e-06-hopper_medium_expert_v2_tatu_mopo",
                122: "critic_num_2_seed_122_0928_015346_pc=1e-06-hopper_medium_expert_v2_tatu_mopo",
                152: "critic_num_2_seed_152_0926_165301_pc=1e-06-hopper_medium_expert_v2_tatu_mopo",
            },
        },
    },
    ##### D4RL - Walker2D #####
    "walker2d-random-v2": {
        "tatu_mopo_sde_disc": {  
            "model": {
                "name": "wk_rand_final",
                "step": -2,
            },
            "config": {
                "RL": 10,
                "CVaR": 0.99,
                "RPC": 0.001,
            },
            "seeds": {
                32: "wk_rand_final_diff=True_cvar=0.99_tdv=disc_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_1118_233555",
                62: "wk_rand_final_diff=True_cvar=0.99_tdv=disc_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_1118_233610",
                92: "wk_rand_final_diff=True_cvar=0.99_tdv=disc_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_1120_144458",
                122: "wk_rand_final_diff=True_cvar=0.99_tdv=disc_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_1120_144505",
            },
        },
        "tatu_mopo_sde_dfd": {  
            "model": {
                "name": "wk_rand_final",
                "step": -2,
            },
            "config": {
                "RL": 10,
                "CVaR": 0.99,
                "RPC": 0.001,
            },
            "seeds": {
                32: "wk_rand_final_diff=True_cvar=0.99_tdv=dad_free_diff_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_1118_233542",
                62: "wk_rand_final_diff=True_cvar=0.99_tdv=dad_free_diff_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_1118_233542",
                92: "wk_rand_final_diff=True_cvar=0.99_tdv=dad_free_diff_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_1120_144438",
                122: "wk_rand_final_diff=True_cvar=0.99_tdv=dad_free_diff_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_1120_144446",
            },
        },
        "tatu_mopo_sde_rew": {  
            "model": {
                "name": "wk_rand_final_rew",
                "step": -2,
            },
            "config": {
                "RL": 10,
                "CVaR": 0.99,
                "RPC": 0.001,
            },
            "seeds": {
                32: "wk_rand_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0923_133736",
                62: "wk_rand_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0923_133740",
                92: "wk_rand_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0923_133743",
                122: "wk_rand_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0923_133747",
                152: "wk_rand_final_rew_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0923_133753",
            },
        },
        "tatu_mopo_sde": {  
            "model": {
                "name": "wk_rand_final",
                "step": -2,
            },
            "config": {
                "RL": 10,
                "CVaR": 0.99,
                "RPC": 0.001,
            },
            "seeds": {
                32: "wk_rand_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0918_210517",
                62: "wk_rand_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0918_210522",
                92: "wk_rand_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0918_210527",
                122: "wk_rand_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0918_210531",
                152: "wk_rand_final_diff=True_cvar=0.99_tdv=diff_density_rl=10_rpc=0.001_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0918_210537",
            },
        },
        "tatu_mopo":{
            "model": {
                "name": "critic_num_2_seed_32_0920_200510_pc=2.0-walker2d_random_v2_tatu_mopo",
            },
            "config": {
                "RL": 5,
                "PC": 2,
                "RPC": 1,
            },
            "seeds": {
                32: "critic_num_2_seed_32_0920_200510_pc=2.0-walker2d_random_v2_tatu_mopo",
                62: "critic_num_2_seed_62_0920_200537_pc=2.0-walker2d_random_v2_tatu_mopo",
                92: "critic_num_2_seed_92_0920_200541_pc=2.0-walker2d_random_v2_tatu_mopo",
                122: "critic_num_2_seed_122_0920_200550_pc=2.0-walker2d_random_v2_tatu_mopo",
                152: "critic_num_2_seed_152_0921_160107_pc=2.0-walker2d_random_v2_tatu_mopo",
            },
        },
        "mopo":{
            "model": {
                "name": "critic_num_2_seed_32_0928_012144_pc=1e-06-walker2d_random_v2_tatu_mopo",
            },
            "config": {
                "RL": 1,
                "RPC": 1,
            },
            "seeds": {
                32: "critic_num_2_seed_32_0928_012144_pc=1e-06-walker2d_random_v2_tatu_mopo",
                62: "critic_num_2_seed_62_0928_012206_pc=1e-06-walker2d_random_v2_tatu_mopo",
                92: "critic_num_2_seed_92_0928_012221_pc=1e-06-walker2d_random_v2_tatu_mopo",
                122: "critic_num_2_seed_122_0928_012228_pc=1e-06-walker2d_random_v2_tatu_mopo",
                152: "critic_num_2_seed_152_0928_012232_pc=1e-06-walker2d_random_v2_tatu_mopo",
            },
        },
    },
    "walker2d-medium-v2": {
        "tatu_mopo_sde_rew": {
            "model": {
                "name": "wk_m_final_rew",
                "step": -2,
            },
            "config": {
                "RL": 10,
                "CVaR": 0.98,
                "RPC": 1,
            },
            "seeds": {
                32: "wk_m_final_rew_diff=True_cvar=0.98_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0923_133804",
                62: "wk_m_final_rew_diff=True_cvar=0.98_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0923_133808",
                92: "wk_m_final_rew_diff=True_cvar=0.98_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0923_133812",
                122: "wk_m_final_rew_diff=True_cvar=0.98_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0923_133815",
                152: "wk_m_final_rew_diff=True_cvar=0.98_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0923_133818",
            },
        },
        "tatu_mopo_sde": {
            "model": {
                "name": "wk_m_final",
                "step": -2,
            },
            "config": {
                "RL": 10,
                "CVaR": 0.98,
                "RPC": 1,
            },
            "seeds": {
                32: "wk_m_final_diff=True_cvar=0.98_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0918_210716",
                62: "wk_m_final_diff=True_cvar=0.98_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0918_210722",
                92: "wk_m_final_diff=True_cvar=0.98_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0918_210729",
                122: "wk_m_final_diff=True_cvar=0.98_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0918_210741",
                152: "wk_m_final_diff=True_cvar=0.98_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0918_210745",
            },
        },
        "tatu_mopo":{
            "model": {
                "name": "critic_num_2_seed_32_0920_200636_pc=2.0-walker2d_medium_v2_tatu_mopo",
            },
            "config": {
                "RL": 5,
                "PC": 2,
                "RPC": 1,
            },
            "seeds": {
                32: "critic_num_2_seed_32_0920_200636_pc=2.0-walker2d_medium_v2_tatu_mopo",
                62: "critic_num_2_seed_62_0920_200642_pc=2.0-walker2d_medium_v2_tatu_mopo",
                92: "critic_num_2_seed_92_0920_200648_pc=2.0-walker2d_medium_v2_tatu_mopo",
                122: "critic_num_2_seed_122_0920_200652_pc=2.0-walker2d_medium_v2_tatu_mopo",
                152: "critic_num_2_seed_152_0921_160131_pc=2.0-walker2d_medium_v2_tatu_mopo",
            },
        },
        "mopo":{
            "model": {
                "name": "critic_num_2_seed_32_0928_012517_pc=1e-06-walker2d_medium_v2_tatu_mopo",
            },
            "config": {
                "RL": 5,
                "RPC": 5,
            },
            "seeds": {
                32: "critic_num_2_seed_32_0928_012517_pc=1e-06-walker2d_medium_v2_tatu_mopo",
                62: "critic_num_2_seed_62_0928_012526_pc=1e-06-walker2d_medium_v2_tatu_mopo",
                92: "critic_num_2_seed_92_0928_012532_pc=1e-06-walker2d_medium_v2_tatu_mopo",
                122: "critic_num_2_seed_122_0928_012542_pc=1e-06-walker2d_medium_v2_tatu_mopo",
                152: "critic_num_2_seed_152_0928_012648_pc=1e-06-walker2d_medium_v2_tatu_mopo",
            },
        },
    },
    "walker2d-medium-replay-v2": {
        "tatu_mopo_sde_rew": {
            "model": {
                "name": "wk_mr_final_rew",
                "step": -2,
            },
            "config": {
                "RL": 10,
                "CVaR": 0.95,
                "RPC": 1,
            },
            "seeds": {
                32: "wk_mr_final_rew_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0923_133246",
                62: "wk_mr_final_rew_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0923_133217",
                92: "wk_mr_final_rew_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0923_133236",
                122: "wk_mr_final_rew_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0923_133252",
                152: "wk_mr_final_rew_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0923_133257",

            },
        },
        "tatu_mopo_sde": {
            "model": {
                "name": "wk_mr_final",
                "step": -2,
            },
            "config": {
                "RL": 10,
                "CVaR": 0.95,
                "RPC": 1,
            },
            "seeds": {
                32: "wk_mr_final_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0919_150650",
                62: "wk_mr_final_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0919_150908",
                92: "wk_mr_final_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0919_150907",
                122: "wk_mr_final_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0919_150911",
                152: "wk_mr_final_diff=True_cvar=0.95_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0919_150916",

            },
        },
        "tatu_mopo":{
            "model": {
                "name": "critic_num_2_seed_32_0921_155758_pc=2.0-walker2d_medium_replay_v2_tatu_mopo",
            },
            "config": {
                "RL": 5,
                "PC": 2,
                "RPC": 1,
            },
            "seeds": {
                32: "critic_num_2_seed_32_0921_155758_pc=2.0-walker2d_medium_replay_v2_tatu_mopo",
                62:  "critic_num_2_seed_62_0921_155801_pc=2.0-walker2d_medium_replay_v2_tatu_mopo",
                92: "critic_num_2_seed_92_0921_155806_pc=2.0-walker2d_medium_replay_v2_tatu_mopo",
                122: "critic_num_2_seed_122_0921_155810_pc=2.0-walker2d_medium_replay_v2_tatu_mopo",
                152: "critic_num_2_seed_152_0921_160217_pc=2.0-walker2d_medium_replay_v2_tatu_mopo",
            },
        },
        "mopo":{
            "model": {
                "name": "critic_num_2_seed_32_0928_012750_pc=1e-06-walker2d_medium_replay_v2_tatu_mopo",
            },
            "config": {
                "RL": 1,
                "RPC": 1,
            },
            "seeds": {
                32: "critic_num_2_seed_32_0928_012750_pc=1e-06-walker2d_medium_replay_v2_tatu_mopo",
                62: "critic_num_2_seed_62_0928_012756_pc=1e-06-walker2d_medium_replay_v2_tatu_mopo",
                92: "critic_num_2_seed_92_0928_012809_pc=1e-06-walker2d_medium_replay_v2_tatu_mopo",
                122: "critic_num_2_seed_122_0928_012816_pc=1e-06-walker2d_medium_replay_v2_tatu_mopo",
                152: "critic_num_2_seed_152_0928_012821_pc=1e-06-walker2d_medium_replay_v2_tatu_mopo",
            },
        },
    },
    "walker2d-medium-expert-v2": {
        "tatu_mopo_sde_disc": {
            "model": {
                "name": "wk_me_final",
                "step": -2,
            },
            "config": {
                "RL": 10,
                "CVaR": 0.98,
                "RPC": 1,
            },
            "seeds": {
                32: "wk_me_final_diff=True_cvar=0.98_tdv=disc_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_1118_140150",
                62: "wk_me_final_diff=True_cvar=0.98_tdv=disc_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_1118_140201",
                92: "wk_me_final_diff=True_cvar=0.98_tdv=disc_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_1120_144228",
                122: "wk_me_final_diff=True_cvar=0.98_tdv=disc_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_1120_144235",
            },
        },
        "tatu_mopo_sde_dfd": {
            "model": {
                "name": "wk_me_final",
                "step": -2,
            },
            "config": {
                "RL": 10,
                "CVaR": 0.98,
                "RPC": 1,
            },
            "seeds": {
                32: "wk_me_final_diff=True_cvar=0.98_tdv=dad_free_diff_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_1118_140120",
                62: "wk_me_final_diff=True_cvar=0.98_tdv=dad_free_diff_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_1118_140120",
                92: "wk_me_final_diff=True_cvar=0.98_tdv=dad_free_diff_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_1120_144316",
                122: "wk_me_final_diff=True_cvar=0.98_tdv=dad_free_diff_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_1120_144312",
            },
        },
        "tatu_mopo_sde_rew": {
            "model": {
                "name": "wk_me_final_rew",
                "step": -2,
            },
            "config": {
                "RL": 10,
                "CVaR": 0.98,
                "RPC": 1,
            },
            "seeds": {
                32: "wk_me_final_rew_diff=True_cvar=0.98_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0923_133432",
                62: "wk_me_final_rew_diff=True_cvar=0.98_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0923_133439",
                92: "wk_me_final_rew_diff=True_cvar=0.98_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0923_133647",
                122: "wk_me_final_rew_diff=True_cvar=0.98_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0923_133443",
                152: "wk_me_final_rew_diff=True_cvar=0.98_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0923_133446",
            },
        },
        "tatu_mopo_sde": {
            "model": {
                "name": "wk_me_final",
                "step": -2,
            },
            "config": {
                "RL": 10,
                "CVaR": 0.98,
                "RPC": 1,
            },
            "seeds": {
                32: "wk_me_final_diff=True_cvar=0.98_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=32_0919_151040",
                62: "wk_me_final_diff=True_cvar=0.98_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=62_0919_151040",
                92: "wk_me_final_diff=True_cvar=0.98_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=92_0919_151029",
                122: "wk_me_final_diff=True_cvar=0.98_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=122_0919_151034",
                152: "wk_me_final_diff=True_cvar=0.98_tdv=diff_density_rl=10_rpc=1.0_rr=0.05_ep=1000_rfq=1000_spe=1000_alr=0.0003_clr=0.0003_seed=152_0919_151038",
            },
        },
        "tatu_mopo":{
            "model": {
                "name": "critic_num_2_seed_32_0921_155850_pc=2.0-walker2d_medium_expert_v2_tatu_mopo",
            },
            "config": {
                "RL": 5,
                "PC": 2,
                "RPC": 1,
            },
            "seeds": {
                32: "critic_num_2_seed_32_0921_155850_pc=2.0-walker2d_medium_expert_v2_tatu_mopo",
                62: "critic_num_2_seed_62_0921_155900_pc=2.0-walker2d_medium_expert_v2_tatu_mopo",
                92: "critic_num_2_seed_92_0921_155906_pc=2.0-walker2d_medium_expert_v2_tatu_mopo",
                122: "critic_num_2_seed_122_0921_155917_pc=2.0-walker2d_medium_expert_v2_tatu_mopo",
                152: "critic_num_2_seed_152_0921_160250_pc=2.0-walker2d_medium_expert_v2_tatu_mopo",
            },
        },
        "mopo":{
            "model": {
                "name": "critic_num_2_seed_32_0928_013105_pc=1e-06-walker2d_medium_expert_v2_tatu_mopo",
            },
            "config": {
                "RL": 1,
                "RPC": 2,
            },
            "seeds": {
                32: "critic_num_2_seed_32_0928_013105_pc=1e-06-walker2d_medium_expert_v2_tatu_mopo",
                62: "critic_num_2_seed_62_0928_013118_pc=1e-06-walker2d_medium_expert_v2_tatu_mopo",
                92: "critic_num_2_seed_92_0928_013122_pc=1e-06-walker2d_medium_expert_v2_tatu_mopo",
                122: "critic_num_2_seed_122_0928_013129_pc=1e-06-walker2d_medium_expert_v2_tatu_mopo",
                152: "critic_num_2_seed_152_0928_013148_pc=1e-06-walker2d_medium_expert_v2_tatu_mopo",
            },
        },
    },
 
    }