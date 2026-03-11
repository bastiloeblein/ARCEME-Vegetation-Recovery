import numpy as np
from typing import List, Dict, Any, Tuple

# Landcover Classes (ESA WorldCover)
ESA_LC_CLASS_MAP: Dict[int, str] = {
    10: "LC_Tree",  # Tree Cover
    20: "LC_Shrub",  # Shrubland
    30: "LC_Grass",  # Grassland
    40: "LC_Crop",  # Cropland
    50: "LC_Settlement",  # Built-up
    60: "LC_Sparse_Veg",  # Bare /sparse vegetation
    70: "LC_Snow",  # Snow and Ice
    80: "LC_Water",  # Permanent water bodies
    90: "LC_Wetland",  # Herbaceous wetland
    100: "LC_Moss_Lichen",  # Moss and/or lichen
}


def landcover_distribution(
    event_data: Any, mapping: Dict[int, str] = ESA_LC_CLASS_MAP
) -> List[Tuple[str, float]]:
    """
    Calculates the fractional distribution of land cover classes from the
    ESA_LC data, converts codes to descriptive names, and returns a list
    sorted in descending order of fractional coverage.

    Args:
        event_data (Any): An object/dataset (e.g., an xarray Dataset)
                          containing the land cover data at event_data.ESA_LC.
        mapping (Dict[int, str]): Dictionary mapping LC codes to class names.

    Returns:
        List[Tuple[str, float]]: A list of tuples (Class Name, Fraction),
                                 sorted by fraction (highest first).
    """
    try:
        # 1. Select the land cover array for the first time step
        # (matching your original logic: .isel(time_esa_worldcover=0))
        lc = event_data.ESA_LC.isel(time_esa_worldcover=0)
    except AttributeError:
        raise ValueError("event_data structure is incorrect or missing 'ESA_LC'.")

    if lc.size == 0:
        return []

    # 2. Calculate the total number of pixels dynamically (more robust)
    total_pixels = lc.size

    # 3. Get unique codes and their counts efficiently using numpy
    unique_codes, counts = np.unique(lc, return_counts=True)

    # 4. Calculate fractional distribution and map names
    distribution_list = []

    for code, count in zip(unique_codes, counts):
        # Calculate fraction and round to 4 decimal places
        fraction = np.round(count * 100 / total_pixels, 2)

        # Get the descriptive name
        class_name = mapping.get(code, f"Unmapped Code ({code})")

        distribution_list.append((class_name, fraction))

    # 5. Sort the list in descending order of fraction (the second element in the tuple)
    sorted_distribution = sorted(distribution_list, key=lambda x: x[1], reverse=True)

    return sorted_distribution
