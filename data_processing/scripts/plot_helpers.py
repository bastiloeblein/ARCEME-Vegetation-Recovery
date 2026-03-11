import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Tuple, Optional, Dict, Any
import folium
from pyproj import CRS, Transformer


## ----------------------------------------------------------------------
## I. Landcover Plot Funktion
## ----------------------------------------------------------------------


# --- ESA WorldCover 10m 2020 Klassifizierung ---
# Die Codes und Farben (in hex) sind Standard für diese Klassifizierung.
LC_MAPPING: Dict[int, str] = {
    10: "Tree cover",  # Dunkelgrün
    20: "Shrubland",  # Helleres Grün
    30: "Grassland",  # Helles leuchtendes Grün
    40: "Cropland",  # Bräunlich/Beige
    50: "Built-up",  # Dunkelgrau
    60: "Bare/Sparse vegetation",  # Mittelgrau
    70: "Snow and ice",  # Sehr helles Weißblau
    80: "Permanent water bodies",  # Kräftiges Blau
    90: "Herbaceous wetland",  # Mittelgrün
    100: "Moss and lichen",  # Silber/Grau
    1: "Unknown/No Data",  # Weiß
}

# Farbwerte (HEX) für die neue Darstellung
LC_COLOR_HEX: Dict[int, str] = {
    10: "#008000",  # Tree cover
    20: "#80C064",  # Shrubland
    30: "#B4FF96",  # Grassland
    40: "#C8AA5A",  # Cropland
    50: "#555555",  # Built-up
    60: "#A9A9A9",  # Bare/Sparse vegetation
    70: "#F0F8FF",  # Snow and ice
    80: "#0000FF",  # Permanent water bodies
    90: "#009664",  # Herbaceous wetland
    100: "#C0C0C0",  # Moss and lichen
    1: "#FFFFFF",  # Unknown/No Data
}
LC_CLASSES = sorted(LC_MAPPING.keys())
LC_COLORS = [LC_COLOR_HEX[code] for code in LC_CLASSES]
LC_LABELS = [LC_MAPPING[code] for code in LC_CLASSES]


def plot_landcover(
    ds: xr.Dataset,
    lc_var: str = "ESA_LC",
    time_dim_lc: str = "time_esa_worldcover",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """
    Plots the Land Cover (LC) map for the first timestep using ESA WorldCover
    classification and creates a custom legend.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset containing the LC variable.
    lc_var : str
        Name of the landcover variable (default 'ESA_LC').
    time_dim_lc : str
        Name of the time dimension for the LC variable (default 'time_esa_worldcover').
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, a new figure is created.
    figsize : tuple
        Matplotlib figure size (only used if ax is None).
    """

    # 1. Data Validation and Subsetting
    if lc_var not in ds:
        print(f"Error: Landcover variable '{lc_var}' not found in the dataset.")
        return

    # Select the first timestep as LC is static
    try:
        lc_snapshot = ds[lc_var].isel({time_dim_lc: 0}).compute()
    except (KeyError, ValueError):
        # Fallback if dimension is missing or already 2D
        lc_snapshot = ds[lc_var].compute()

    # 2. Plotting Setup
    standalone = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        standalone = True

    # 3. Create Colormap and Normalization
    cmap = mcolors.ListedColormap(LC_COLORS)
    bounds = np.array(LC_CLASSES)
    bounds = np.insert(bounds, 0, bounds[0] - 5)
    bounds = bounds[:-1] + np.diff(bounds) / 2
    bounds = np.append(bounds, bounds[-1] + 10)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # 4. Plot Landcover Map
    ax.imshow(lc_snapshot.values, cmap=cmap, norm=norm, interpolation="nearest")

    # 5. Create Legend (Using Patch objects to map colors)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=cmap(norm(code))) for code in LC_CLASSES]

    ax.legend(
        handles,
        LC_LABELS,
        title="ESA WorldCover Classes",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=10,
    )

    # 6. Formatting
    ax.set_title(f"Landcover Map ({lc_var})", fontsize=14)
    ax.axis("off")  # Disable axis ticks for better visualization

    if standalone:
        plt.tight_layout()
        plt.show()


## ----------------------------------------------------------------------
## V. Open Street Maps Plot Funktion
## ----------------------------------------------------------------------


def show_data_on_interactive_map(ds, edge_size_m=10000.0):
    """
    Transforms UTM coordinates to WGS84 and displays the location on an interactive map.
    The function provides both a detailed street map (with borders/cities) and a
    satellite view, which the user can switch between using the Layer Control.

    Args:
        ds (xarray): Data set
        edge_size_m (float): The side length of the bounding box in meters.

    Returns:
        folium.Map: A Folium map object.
    """

    # 1. Get information from ds
    epsg_code = ds.epsg
    central_x = ds.central_x
    central_y = ds.central_y

    # 2. Transform UTM (e.g., EPSG:32634) to WGS84 (EPSG:4326)
    # WGS84 (4326) is the standard Latitude/Longitude system used by web maps.
    transformer = Transformer.from_crs(CRS(epsg_code), CRS(4326))
    latitude, longitude = transformer.transform(central_x, central_y)

    # 2. Create the Folium Map
    # Start with 'OpenStreetMap', which provides rich context (borders, cities, roads).
    m = folium.Map(location=[latitude, longitude], zoom_start=13, tiles="OpenStreetMap")

    # Add a Satellite tile layer as an alternative option.
    # The user can click the Layer Control icon (top right of the map) to switch views.
    folium.TileLayer("Esri.WorldImagery", name="Satellite View").add_to(m)

    # 3. Add the central marker
    folium.Marker(
        [latitude, longitude],
        tooltip="Central Point",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(m)

    # 4. Calculate and add the Bounding Box
    half_size = edge_size_m / 2

    # Calculate corners in UTM
    utm_corners = [
        (central_x - half_size, central_y - half_size),
        (central_x + half_size, central_y + half_size),
    ]

    # Transform UTM corners for the bounding box (bottom-left and top-right)
    lat_min, lon_min = transformer.transform(utm_corners[0][0], utm_corners[0][1])
    lat_max, lon_max = transformer.transform(utm_corners[1][0], utm_corners[1][1])

    # Add the bounding box as a Rectangle
    folium.Rectangle(
        bounds=[[lat_min, lon_min], [lat_max, lon_max]],
        color="yellow",
        weight=3,
        fill=False,
        tooltip=f"Data Coverage Area ({edge_size_m/1000}km x {edge_size_m/1000}km)",
    ).add_to(m)

    # 5. Add a Layer Control to enable switching between base map layers
    folium.LayerControl().add_to(m)

    return m


## ----------------------------------------------------------------------
## V. Find Cloud free indices for S2
## ----------------------------------------------------------------------


def find_cloud_free_indices(ds, threshold=0.99):
    """
    Findet die Indizes der Sentinel-2 Zeitschritte, die nahezu wolkenfrei sind.
    Basiert auf der 'mask_complete_quality' Variable.
    """
    # 1. Räumliches Mittel berechnen (Qualitätsrate pro Zeitschritt)
    quality_means = ds.mask_complete_quality.mean(dim=("x", "y"))

    # 2. Indizes finden, die die Bedingung erfüllen
    indices = np.where(quality_means > threshold)[0]

    # 3. Anzahl berechnen
    count = len(indices)

    print(f"Gefundene wolkenfreie Zeitschritte (> {threshold*100}%): {count}")

    return indices


##
## VI. Plot Mean variable over temporal dimension
##


def plot_spatial_analysis_map(
    ds: Any,
    variable_name: str,
    time_dim: str,
    title: str = "Spatial Analysis Map",
    cmap_name: str = "viridis",
    smart_scaling: bool = False,
) -> None:
    """
    Plots a 2D map showing the mean of a variable over the entire time series.
    Applies smart scaling based on IQR to ensure the plot is not dominated by outliers.

    Args:
        ds (Any): The xarray Dataset.
        variable_name (str): Name of the variable to analyze (e.g., 'missing_overall', 'vv_db').
        time_dim (str): Name of the time dimension (e.g., 'time_sentinel_2_l2a').
        title (str): Title of the plot.
        cmap_name (str): Colormap name. Uses 'RdYlGn_r' for fractions (0-1).
        smart_scaling (bool): If True, uses the IQR method (Q5 and Q95) for robust color limits.
    """

    # 1. Berechne die räumliche Karte (Mittelwert über die Zeitdimension)
    # Dies ist die 2D-Karte, die geplottet wird.
    spatial_map = ds[variable_name].mean(dim=time_dim)

    # 2. Intelligente Skalierung (IQR-Methode)
    # Wenn die Werte bereits eine Fraktion sind (0-1), behalten wir die Skala bei.
    if (
        spatial_map.max().values <= 1.0
        and spatial_map.min().values >= 0.0
        and not smart_scaling
    ):
        vmin = 0.0
        vmax = 1.0
        cmap_name = "RdYlGn_r"
    else:
        # Führe eine robuste Skalierung durch, um Ausreißer zu ignorieren
        # Dies ist das "smarte" Skalieren: Schließe die extremsten 5% der Daten aus.
        if smart_scaling or not (
            spatial_map.max().values <= 1.0 and spatial_map.min().values >= 0.0
        ):
            # Entferne NaNs für die Quantilberechnung
            data_valid = spatial_map.values[~np.isnan(spatial_map.values)]

            # Verwende das 5. und 95. Perzentil als Farb-Grenzen
            vmin = np.percentile(data_valid, 5)
            vmax = np.percentile(data_valid, 95)
        else:
            # Fallback auf die tatsächlichen Min/Max-Werte
            vmin = spatial_map.min().values
            vmax = spatial_map.max().values

    # 3. Plotting der 2D-Karte
    fig, ax = plt.subplots(figsize=(9, 8))

    plot_handle = spatial_map.plot.imshow(
        ax=ax,
        cmap=cmap_name,
        vmin=vmin,
        vmax=vmax,
        cbar_kwargs={
            "label": f"Temporal Mean of {variable_name} ({vmin:.2f} to {vmax:.2f})",
            "extend": "both" if smart_scaling else "neither",
        },
    )

    plot_handle.colorbar.set_ticks([vmin, (vmin + vmax) / 2, vmax])

    # Titel und Labels
    ax.set_title(f"{title}: Mean of {variable_name} over Time", fontsize=10)
    ax.set_xlabel("X-Coordinate", fontsize=10)
    ax.set_ylabel("Y-Coordinate", fontsize=10)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.show()
