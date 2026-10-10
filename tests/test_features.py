import pandas as pd

from app.features import N_FEATURES, build_features


def test_shape_and_derived():
    df = pd.DataFrame([{"type": "M", "air_temp_k": 300.0, "process_temp_k": 310.0, "rotational_speed_rpm": 60.0,
                        "torque_nm": 10.0, "tool_wear_min": 5.0}])
    X = build_features(df)
    assert X.shape == (1, N_FEATURES)
    assert X[0, 1] == 1 and X[0, 0] == 0          # one-hot M
    assert abs(X[0, 8] - 10.0) < 1e-4              # temp diff
    assert abs(X[0, 9] - 10 * 2 * 3.14159265) < 1e-2  # power = torque * rad/s
