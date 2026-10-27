import yaml

from garage_recon import config as config_lib


def test_default_imu_grade_keeps_the_consumer_priors():
    cfg = config_lib.load_config()
    assert cfg.imu_grade == "consumer"
    assert cfg.imu_grade_applied
    assert cfg.backend.gyro_bias_prior == 0.01
    assert cfg.backend.accel_bias_prior == 0.2
    assert cfg.odometry.lio.init_accel_bias_sigma == 0.1


def test_imu_grade_fills_only_what_the_user_does_not_set(tmp_path):
    user = tmp_path / "user.yaml"
    user.write_text("imu_grade: tactical\nbackend: {gyro_bias_prior: 0.004}\n")
    cfg = config_lib.load_config(user)
    assert cfg.backend.gyro_bias_prior == 0.004  # Explicit wins.
    assert cfg.backend.accel_bias_prior == 0.03
    assert cfg.odometry.lio.init_gyro_bias_sigma == 0.002
    # A saved, resolved configuration loaded again keeps its values.
    saved = tmp_path / "config.used.yaml"
    data = cfg.to_dict()
    data["backend"]["accel_bias_prior"] = 0.5
    saved.write_text(yaml.safe_dump(data))
    assert config_lib.load_config(saved).backend.accel_bias_prior == 0.5
