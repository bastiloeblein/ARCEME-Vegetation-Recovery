

def print_channel_info(
    s2, s1, era5, stat, expected_input_channels=None, include_landcover=True
):
    if not s2 or s2[0] != "kNDVI":
        raise ValueError("The first S2 channel must be kNDVI.")

    print("\n" + " channel mapping ".center(50, "="))
    current_idx = 0

    # Shared features: same positions in context and future input
    print(f"CH {current_idx:02d}: S2_kNDVI")
    current_idx += 1

    for v in era5:
        print(f"CH {current_idx:02d}: ERA5_{v}")
        current_idx += 1

    if include_landcover:
        print(
            f"CH {current_idx:02d} - {current_idx + 11:02d}: " "ESA_Landcover (One-Hot)"
        )
        current_idx += 12

    for v in stat:
        if v != "ESA_LC":
            print(f"CH {current_idx:02d}: Static_{v}")
            current_idx += 1

    # Context-only features
    for v in s2[1:]:
        print(f"CH {current_idx:02d}: S2_{v}")
        current_idx += 1

    for v in s1:
        print(f"CH {current_idx:02d}: S1_{v}")
        current_idx += 1

    print(f"CH {current_idx:02d}: Mask_kNDVI")
    current_idx += 1

    if len(s2) > 1:
        print(f"CH {current_idx:02d}: Mask_S2")
        current_idx += 1

    if s1:
        print(f"CH {current_idx:02d}: Mask_S1")
        current_idx += 1

    if expected_input_channels is not None:
        assert current_idx == expected_input_channels, (
            f"Printed {current_idx} channels, but model expects "
            f"{expected_input_channels}."
        )

    print(f"Total context channels: {current_idx}")
    print("=" * 50 + "\n")
