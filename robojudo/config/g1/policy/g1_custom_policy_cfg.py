from robojudo.policy.policy_cfgs import CustomPolicyCfg


class G1CustomPolicyCfg(CustomPolicyCfg):
    robot: str = "g1"
    model_dir: str = "assets/models/g1/custom/exported_1"
    model_backend: str = "onnx"
    # policy_name: str = "policy_wo_gait"
    policy_name: str = "policy_0"

    # model_suffix: str = ".pt"
    robot_config_file: str = "assets/models/g1/custom/exported_1/robot_config.yaml"


class G1CustomPolicy2Cfg(CustomPolicyCfg):
    robot: str = "g1"
    model_dir: str = "assets/models/g1/custom/exported"
    model_backend: str = "torchscript"
    policy_name: str = "policy_wo_gait"
    model_suffix: str = ".pt"
    robot_config_file: str = "assets/models/g1/custom/exported/robot_config.yaml"
