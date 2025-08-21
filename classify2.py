import streamlit as st
import rasterio
import geopandas as gpd
import numpy as np
from rasterstats import zonal_stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from rasterio.plot import reshape_as_image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

st.set_page_config(page_title="GeoHackers Land Classification", layout="wide")
st.title(" 🛰️ Land Use Classification and Visualization Tool")

# --- File uploads ---
st.sidebar.header(" Upload your data")
raster_file = st.sidebar.file_uploader(" Upload satellite raster (GeoTIFF)", type=["tif", "tiff"])
shapefile_file = st.sidebar.file_uploader(" Upload training shapefile (.zip)", type=["zip"])

if raster_file and shapefile_file:
    import tempfile
    import zipfile
    import os

    with st.spinner("⏳ Processing data..."):
        with tempfile.TemporaryDirectory() as tmpdir:
            shapefile_zip_path = os.path.join(tmpdir, "training.zip")
            with open(shapefile_zip_path, "wb") as f:
                f.write(shapefile_file.getbuffer())
            with zipfile.ZipFile(shapefile_zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdir)

            shp_path = next((os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.endswith(".shp")), None)
            if not shp_path:
                st.error(" No .shp file found in the uploaded zip.")
                st.stop()

            gdf = gpd.read_file(shp_path)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_raster:
                tmp_raster.write(raster_file.read())
                raster_path = tmp_raster.name

            gdf.columns = gdf.columns.str.strip().str.lower()
            possible_class_columns = ['classname', 'classvalue', 'class', 'class_value', 'label', 'landuse', 'type']
            class_column = next((col for col in possible_class_columns if col in gdf.columns), None)
            if class_column is None:
                st.error(f" Class column not found! Available columns: {list(gdf.columns)}")
                st.stop()

            st.success(f" Using '{class_column}' column as class labels.")

            with rasterio.open(raster_path) as src:
                raster_crs = src.crs
                img = src.read()
                meta = src.meta.copy()
                nodata_value = src.nodatavals[0] if src.nodatavals else None  # Get no-data value
                mask = src.read_masks(1)  # Get mask for valid data (1=valid, 0=invalid)

            gdf = gdf.to_crs(raster_crs)

            # Zonal statistics
            X, y = [], []
            for i in range(len(gdf)):
                means = []
                for band in range(1, img.shape[0] + 1):
                    stats = zonal_stats(gdf.iloc[[i]], raster_path, stats=["mean"], band=band, all_touched=True)
                    means.append(stats[0]['mean'])
                X.append(means)
                y.append(gdf.iloc[i][class_column])

            X = np.array(X)
            y = np.array(y)

            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X, y_encoded)

            # Classify all pixels
            rows, cols = img.shape[1], img.shape[2]
            reshaped = reshape_as_image(img).reshape(-1, img.shape[0])
            reshaped = np.nan_to_num(reshaped, nan=0)
            predictions = clf.predict(reshaped)
            classified = predictions.reshape(rows, cols)

            # Apply mask to classified output
            if nodata_value is not None:
                # Create a masked array where no-data pixels are set to NaN
                classified = np.ma.masked_array(classified, mask=~mask.astype(bool))

        # Show RGB image
        with st.expander("📷 Original RGB Image"):
            st.subheader("Satellite RGB View")
            rgb_img = np.dstack([img[0], img[1], img[2]])
            rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
            st.image(rgb_img, use_column_width=True)

        # Custom color palette
        custom_colors = [
            "#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]
        num_classes = len(le.classes_)
        cmap = mcolors.ListedColormap(custom_colors[:num_classes])

        # Plot classified land use
        st.subheader("🗺️ Classified Land Use Map")
        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(classified, cmap=cmap, interpolation='none')
        ax.set_title("Predicted Land Use Classes", fontsize=14)
        ax.axis('off')

        # Legend
        handles = [plt.Rectangle((0, 0), 1, 1, color=custom_colors[i]) for i in range(num_classes)]
        ax.legend(handles, le.classes_, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title="Land Classes")

        st.pyplot(fig)

        st.success("✅ Classification complete!")

else:

    st.warning("📌 Please upload both raster and training shapefile (as .zip).")
