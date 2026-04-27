from robojudo.policy.policy_cfgs import ParkourPolicyCfg


class G1ParkourPolicyCfg(ParkourPolicyCfg):
    robot: str = "g1"
    policy_name: str = "parkour_policy_merged"
    robot_config_file: str = "assets/models/g1/parkour/robot_config.yaml"

