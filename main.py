import streamlit as st
import cv2
import numpy as np
from config import (
    TILE_SIZE, DEFAULT_HIGH_RANK, DEFAULT_LOW_RANK, DEFAULT_HIGH_QUALITY, DEFAULT_LOW_QUALITY,
    DEFAULT_HIGH_PCA, DEFAULT_LOW_PCA, DEFAULT_HIGH_K, DEFAULT_LOW_K, DEFAULT_HIGH_WAVELET, DEFAULT_LOW_WAVELET,
    ALGORITHM_OPTIONS
)
from utils.detectors import detect_faces
from utils.image import (
    compress_image_by_tiles, compress_image_pil, overlay_rectangles
)
from utils.entropy import generate_rank_map
from PIL import Image as PILImage
from io import BytesIO
from streamlit_drawable_canvas import st_canvas
from streamlit_overlay import heatmap_overlay
from sklearn.cluster import KMeans # Safe KMeans single-tile compressor 

def compress_tile_kmeans(tile, n_clusters=4):
    flat = tile.flatten().reshape(-1, 1)
    safe_k = min(max(2, n_clusters), flat.shape[0], len(np.unique(flat)), 8)  # cap at 8 for speed/safety
    if safe_k < 1:
        safe_k = 1
    try:
        model = KMeans(n_clusters=safe_k, n_init=1, random_state=42)
        model.fit(flat)
        centers = model.cluster_centers_
        labels = model.labels_
        compressed_flat = centers[labels].reshape(tile.shape)
        return compressed_flat.astype(np.uint8)
    except Exception:
        return tile

def get_image_bytes(img_np_arr, fmt='JPEG'):
    img = PILImage.fromarray(img_np_arr)
    buf = BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf

def is_in_important_region(x, y, tile_size, regions):
    for (x1, y1, x2, y2) in regions:
        if x < x2 and x + tile_size > x1 and y < y2 and y + tile_size > y1:
            return True
    return False

st.set_page_config(page_title="Smart Split Image Compressor 🗜️", layout="wide")
st.title("🖼️ Smart Split Image Compressor 🗜️")
st.subheader("✨ Explore AI-powered compression algorithms: SVD, Autoencoder, Wavelet, KMeans and more")

if 'visit_count' not in st.session_state:
    st.session_state.visit_count = 1
else:
    st.session_state.visit_count += 1

st.caption(
    f"👋 This web app has run **{st.session_state.visit_count}** times. "
    "Share it and let the count grow! 🚀"
)

if "canvas_objects" not in st.session_state:
    st.session_state.canvas_objects = []
if "manual_regions" not in st.session_state:
    st.session_state.manual_regions = []

MAX_DIM = 1024  # Downscale limit for memory safety

uploaded = st.file_uploader(
    "📤 Upload an image 🖼️", type=["jpg", "jpeg", "png"]
)
if uploaded:
    img_array = np.frombuffer(uploaded.read(), np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if image is None:
        st.error("❌ Failed to decode image.")
        st.stop()

    h, w = image.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        st.info(f"Image was downscaled to {image.shape[1]}x{image.shape[0]} to prevent memory errors.")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    algo_selected = st.selectbox("🧮 Select Compression Algorithm", ALGORITHM_OPTIONS, index=0)
    if algo_selected == "SVD":
        high_param, low_param = DEFAULT_HIGH_RANK, DEFAULT_LOW_RANK
    elif algo_selected == "DCT":
        high_param, low_param = DEFAULT_HIGH_QUALITY, DEFAULT_LOW_QUALITY
    elif algo_selected == "PIL":
        high_param, low_param = DEFAULT_HIGH_QUALITY, DEFAULT_LOW_QUALITY
    elif algo_selected == "PCA":
        high_param = int(DEFAULT_HIGH_PCA * 100)
        low_param = int(DEFAULT_LOW_PCA * 100)
    elif algo_selected == "KMeans":
        high_param, low_param = DEFAULT_HIGH_K, DEFAULT_LOW_K
    elif algo_selected == "Wavelet":
        high_param, low_param = DEFAULT_HIGH_WAVELET, DEFAULT_LOW_WAVELET
    elif algo_selected == "Autoencoder":
        high_param, low_param = 0, 0
    else:
        high_param, low_param = 25, 5

    detect_faces_option = st.checkbox("🤖 Auto-detect faces (ROI)", value=True)
    manual_regions_option = st.checkbox("✍️ Manually define important regions", value=True)
    entropy_option = st.checkbox("🧠 Use entropy-based adaptive compression", value=False)
    compression_percent = st.slider(
        "🔧 Compression Level (higher = more aggressive on unimportant regions)",
        min_value=10, max_value=100, value=70, step=5
    )
    low_param_setting = int(high_param * (100 - compression_percent) / 100)
    if algo_selected == "PIL":
        low_param_setting = max(1, min(100, low_param_setting))
    else:
        low_param_setting = max(0, min(high_param, low_param_setting))

    st.markdown("---")
    st.header("1️ Original Image and ROI Selection")
    st.image(image, channels="BGR", caption="Original Image 🖼️", use_container_width=True)

    face_regions = []
    if detect_faces_option:
        face_regions = detect_faces(image)
        if face_regions:
            st.success(f"✅ {len(face_regions)} face(s) detected.")
        else:
            st.info("No faces detected.")

    # ROI Drawing Canvas, with mode-dependent instructions
    drawing_mode_ui = None
    if manual_regions_option:
        tips = {
            "rectangle": "Draw by dragging. Release to finish each rectangle.",
            "brush": "Draw freeform. Large/fast strokes may slow the UI."
        }
        drawing_mode_ui = st.selectbox("🛠️ Choose Drawing Tool", ["rectangle", "brush"], index=0)
        st.markdown(tips[drawing_mode_ui])

        mode_mapping = {"rectangle": "rect", "brush": "freedraw"}
        drawing_mode = mode_mapping[drawing_mode_ui]

        max_canvas_width = min(image.shape[1], MAX_DIM)
        canvas_width = max_canvas_width
        scale_ratio = canvas_width / image.shape[1]
        canvas_height = int(image.shape[0] * scale_ratio)
        pil_image = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        resized_pil = pil_image.resize((canvas_width, canvas_height))
        last_objs = st.session_state.get("last_canvas_objects", [])
        
        assert resized_pil is None or isinstance(resized_pil, PILImage.Image)
        canvas_result = st_canvas(
            fill_color="rgba(0,255,0,0.12)",
            stroke_width=2,
            stroke_color="#1EDD00",
            background_image=resized_pil, # type: ignore
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode=drawing_mode,
            initial_drawing={"objects": st.session_state.canvas_objects, "version": "4.4.0"},
            key="main_canvas"
        )

        manual_regions = []
        if canvas_result.json_data and "objects" in canvas_result.json_data:
            new_objs = canvas_result.json_data["objects"]
            # Only update if changed to avoid stream of events
            if new_objs != last_objs:
                for obj in new_objs:
                    # Rectangle
                    if obj.get("type") == "rect":
                        left = int(obj["left"] / scale_ratio)
                        top = int(obj["top"] / scale_ratio)
                        width = int(obj["width"] / scale_ratio)
                        height = int(obj["height"] / scale_ratio)
                        manual_regions.append((left, top, left + width, top + height))
                    # Polygon/Freehand: Use bounding box
                    elif obj.get("type") in ("polygon", "freedraw"):
                        xs = [int(x / scale_ratio) for x, y in obj.get("path", [])]
                        ys = [int(y / scale_ratio) for x, y in obj.get("path", [])]
                        if xs and ys:
                            manual_regions.append((min(xs), min(ys), max(xs), max(ys)))
                    # Line
                    elif obj.get("type") == "line":
                        x1 = int(obj["x1"] / scale_ratio)
                        y1 = int(obj["y1"] / scale_ratio)
                        x2 = int(obj["x2"] / scale_ratio)
                        y2 = int(obj["y2"] / scale_ratio)
                        manual_regions.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
                st.session_state.canvas_objects = new_objs
                st.session_state.manual_regions = manual_regions
                st.session_state.last_canvas_objects = list(new_objs)
        manual_regions = st.session_state.get("manual_regions", [])
    else:
        manual_regions = []

    # Merged all regions for compression/overlays
    all_regions = []
    if face_regions:
        all_regions.extend(face_regions)
    if manual_regions:
        all_regions.extend(manual_regions)

    # Compression 
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray_image.shape

    if algo_selected == 'PIL':
        rank_map = None
    elif entropy_option and high_param > 0:
        rank_map = generate_rank_map(gray_image, TILE_SIZE, min_rank=low_param_setting, max_rank=high_param)
    else:
        tiles_x = h // TILE_SIZE
        tiles_y = w // TILE_SIZE
        rank_map = []
        for i in range(tiles_x):
            row = []
            for j in range(tiles_y):
                y, x = i * TILE_SIZE, j * TILE_SIZE
                if is_in_important_region(x, y, TILE_SIZE, all_regions):
                    row.append(high_param)
                else:
                    row.append(low_param_setting)
            rank_map.append(row)

    # Display overlays and compress 
    show_rectangles = st.checkbox("🟩 Overlay ROIs on images", value=True)
    show_heatmap = st.checkbox("🌡️ Show compression quality heatmap", value=True)
    regions_for_overlay = all_regions
    original_for_viz = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_overlay = overlay_rectangles(original_for_viz, regions_for_overlay) if show_rectangles else original_for_viz

    st.image(original_overlay, caption="Original + ROI overlay", use_container_width=True)

    with st.spinner("🗜️ Compressing image..."): 
        compressed = None
        compressed_overlay = None

        if algo_selected == 'PIL':
            compressed = compress_image_pil(gray_image, quality=low_param_setting)
            compressed_overlay = cv2.cvtColor(compressed, cv2.COLOR_GRAY2RGB)

        elif algo_selected == "PCA":
            # ensuring low_param_setting in (0.01, 0.99)
            percent = (100 - compression_percent) / 100
            low_param_setting = max(0.01, min(0.99, float(high_param) * percent))
            # ensuring compress_image_by_tiles uses low_param_setting (float) for PCA tiles
            compressed = compress_image_by_tiles(gray_image, rank_map, algorithm=algo_selected)  # NO pca_variance kwarg
            compressed_overlay = cv2.cvtColor(compressed, cv2.COLOR_GRAY2RGB)

        elif algo_selected == "KMeans":
            tile_h, tile_w = TILE_SIZE, TILE_SIZE
            out = np.zeros_like(gray_image)
            tiles_x = gray_image.shape[0] // tile_h
            tiles_y = gray_image.shape[1] // tile_w
            for i in range(tiles_x):
                for j in range(tiles_y):
                    y1, y2 = i * tile_h, (i + 1) * tile_h
                    x1, x2 = j * tile_w, (j + 1) * tile_w
                    tile = gray_image[y1:y2, x1:x2]
                    if rank_map is not None:
                        raw_k = rank_map[i][j]
                    else:
                        raw_k = low_param_setting
                    k = min(max(2, raw_k), tile.size, len(np.unique(tile)), 8)
                    tile_compressed = compress_tile_kmeans(tile, n_clusters=k)
                    out[y1:y2, x1:x2] = tile_compressed
            compressed = out
            compressed_overlay = cv2.cvtColor(compressed, cv2.COLOR_GRAY2RGB)

        elif algo_selected in {"SVD", "DCT", "Wavelet", "Autoencoder"}:
            # All other valid algorithms (tile compressor defined on some other files)
            compressed = compress_image_by_tiles(gray_image, rank_map, algorithm=algo_selected)
            compressed_overlay = cv2.cvtColor(compressed, cv2.COLOR_GRAY2RGB)

        else:
            raise ValueError(f"Unknown/unsupported algorithm selected: {algo_selected}")

        if show_rectangles:
            compressed_overlay = overlay_rectangles(compressed_overlay, regions_for_overlay, color=(255, 30, 30))
            st.image(compressed_overlay, caption=f"Compressed ({algo_selected}) with overlays", use_container_width=True)


    if show_heatmap and algo_selected != 'PIL' and rank_map is not None:
        tiles_x = gray_image.shape[0] // TILE_SIZE
        tiles_y = gray_image.shape[1] // TILE_SIZE
        error_map = np.zeros((tiles_x, tiles_y), dtype=np.float32)
        for i in range(tiles_x):
            for j in range(tiles_y):
                y1, y2 = i * TILE_SIZE, (i + 1) * TILE_SIZE
                x1, x2 = j * TILE_SIZE, (j + 1) * TILE_SIZE
                orig_tile = gray_image[y1:y2, x1:x2]
                comp_tile = compressed[y1:y2, x1:x2]
                if orig_tile.shape == comp_tile.shape:
                    mse = np.mean((orig_tile.astype(np.float32) - comp_tile.astype(np.float32)) ** 2)
                    error_map[i, j] = mse

        norm_error_map = (error_map - error_map.min()) / (np.ptp(error_map) + 1e-6)
        heatmap = cv2.resize(norm_error_map.astype(np.float32), (compressed_overlay.shape[1], compressed_overlay.shape[0]), interpolation=cv2.INTER_NEAREST)
        try:
            heatmap_overlay(images=compressed_overlay, masks=heatmap, colormap=cv2.COLORMAP_JET, toggle_label="Display heatmap overlay")
        except Exception as e:
            st.warning(f"Could not display heatmap overlay: {e}")

    st.download_button(
        label="💾 Download Compressed Image",
        data=get_image_bytes(compressed_overlay),
        file_name=f"compressed_output_{algo_selected}.jpg",
        mime="image/jpeg"
    )
else:
    st.info("Upload an image to get started!")