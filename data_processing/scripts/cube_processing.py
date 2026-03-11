from datetime import timedelta
import pandas as pd


def add_event_metadata(ds, df_data, cube_id):
    """
    Extrahiert Ereignisdaten aus einem Dataframe und fügt sie als
    Attribute (Metadaten) zum xarray-Dataset hinzu.
    """
    # 1. Daten für die spezifische Cube ID filtern
    if "cube_id" not in ds.attrs:
        ds.attrs["cube_id"] = cube_id

    current_id = ds.attrs["cube_id"]
    cube_data = df_data[df_data["DisNo."] == current_id]

    if cube_data.empty:
        print(f"Warnung: Keine Daten für Cube {cube_id} in df_data gefunden.")
        return ds

    # 2. Zeitstempel extrahieren (als Skalare mit .iloc[0])
    try:
        p_start = pd.to_datetime(cube_data["start_date"].iloc[0])
        p_end = pd.to_datetime(cube_data["end_date"].iloc[0])

        # Zeitfenster berechnen
        d_start = p_start - timedelta(days=40)
        d_end = p_start - timedelta(days=10)

        # Als Strings speichern (verhindert den "Not JSON serializable" Fehler bei to_zarr)
        ds.attrs["precip_start_date"] = p_start.strftime("%Y-%m-%d")
        ds.attrs["precip_end_date"] = p_end.strftime("%Y-%m-%d")
        ds.attrs["drought_start_date"] = d_start.strftime("%Y-%m-%d")
        ds.attrs["drought_end_date"] = d_end.strftime("%Y-%m-%d")

        # Hilfreiche Info für die Konsole
        print(f"Metadaten für {cube_id} erfolgreich hinzugefügt.")
        print(
            f"  Event: {ds.attrs['precip_start_date']} bis {ds.attrs['precip_end_date']}"
        )

    except Exception as e:
        print(f"Fehler beim Verarbeiten der Daten für {cube_id}: {e}")

    return ds
