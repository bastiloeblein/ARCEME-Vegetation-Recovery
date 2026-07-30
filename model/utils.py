import zarr
import numpy as np

def print_channel_info(s2, s1, era5, stat, expected_input_channels=None):
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

    print(
        f"CH {current_idx:02d} - {current_idx + 11:02d}: "
        "ESA_Landcover (One-Hot)"
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


# def print_channel_info(s2, s1, era5, stat):
#     print("\n" + " channel mapping ".center(40, "="))
#     current_idx = 0

#     # Sentinel-2
#     for v in s2:
#         print(f"CH {current_idx:02d}: S2_{v}")
#         current_idx += 1
#     # Sentinel-1
#     for v in s1:
#         print(f"CH {current_idx:02d}: S1_{v}")
#         current_idx += 1
#     # ERA5
#     for v in era5:
#         print(f"CH {current_idx:02d}: ERA5_{v}")
#         current_idx += 1
#     # Masks
#     print(f"CH {current_idx:02d}: Mask_S2")
#     print(f"CH {current_idx+1:02d}: Mask_S1")
#     current_idx += 2
#     # Static
#     print(f"CH {current_idx:02d} - {current_idx+11:02d}: ESA_Landcover (One-Hot)")
#     current_idx += 12
#     for v in stat:
#         if v != "ESA_LC":  # ESA_LC ist schon drin
#             print(f"CH {current_idx:02d}: Static_{v}")
#             current_idx += 1
#     print("=" * 40 + "\n")


# def get_cloud_stats_zarr(file_list):
#     """Scannt Zarr-Cubes und berechnet den Anteil maskierter Pixel."""
#     total_pixels = 0
#     masked_pixels = 0
    
#     for f in file_list:
#         try:
#             # Öffne den Zarr-Store (nur Read-Mode)
#             store = zarr.open(f, mode='r')
            
#             # Wir nehmen an, deine Maske liegt unter dem Key 'mask' 
#             # oder ist Teil eines Arrays. Passe den Namen ggf. an ('mask_s2' etc.)
#             if 'mask' in store:
#                 mask_data = store['mask'][:] # Lade nur die Maske in den RAM
#                 total_pixels += mask_data.size
#                 # Wir zählen alles, was 0 ist (Invalid/Cloud)
#                 masked_pixels += np.sum(mask_data == 0)
#         except Exception as e:
#             print(f"⚠️ Konnte Statistik für {f} nicht lesen: {e}")
            
#     if total_pixels == 0: return 0.0
#     return (masked_pixels / total_pixels) * 100