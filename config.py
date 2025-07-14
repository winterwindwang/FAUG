
attack_config = {
    "mifgsm": {

    },
    "dim": {

    },
    "vnifgsm": {

    },
    "vmifgsm": {

    },
    "smm": {  # s^2I-FGSM: ECCV2022 [Frequency domain model augmentation for adversarial attack]
        "num_spectrum": 5,
    },
    "admix": { # Admix: ICCV2021 [Enhancing the transferability of adversarial attacks]
        "num_scale": 5,
        "num_admix": 3,
        "admix_strength": 0.3,
    },
    "naa": { # naa: CPVR2022 [Improving adversarialtransferability via neuron attribution-based attacks]
        "N": 5, # default is 20
    },
    "taig": { # TAIG: ICLR2022 [Transferable adversarial attack based on integrated gradients]
        "steps": 5,
    },
    "cwa": { # CWA: ICLR2024 [Rethinking Model Ensemble in Transfer-based Adversarial Attacks]

    },
    "LinBP ": { # LinBP: NeuroIPS2020 [Backpropagating linearly improves transferability of adversarial examples]

    },
    "SAM-JR": { # SAM-JR: SP2024 [Why Does Little Robustness Help? Understanding Adversarial Transferability From Surrogate Training]

    }
}