import streamlit as st
import cv2
import numpy as np
import hmac
import os
from dotenv import load_dotenv
import math
import pandas as pd
from skimage.filters import frangi
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
import time
from fpdf import FPDF
import tempfile
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load the .env
load_dotenv()


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        if hmac.compare_digest(
            st.session_state["password"], os.environ.get("STREAMLIT_PASSWORD", "")
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()

st.set_page_config(layout="wide", page_title="Fiber Analysis Pro")


# --- CORE IMAGE PROCESSING ENGINE ---
def process_fiber_image(image_bytes, params):
    """
    Process fiber image with configurable parameters.

    params dict should contain:
        - fiber_core_threshold: threshold for Frangi output (default 20)
        - min_object_area: minimum area for noise filtering (default 10)
        - blob_threshold: threshold for blob identification (default 21)
        - dist_transform_max: max distance transform value (default 10)
    """
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_original = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # Extract parameters with defaults
    fiber_core_threshold = params.get('fiber_core_threshold', 20)
    min_object_area = params.get('min_object_area', 10)
    blob_threshold = params.get('blob_threshold', 21)
    dist_transform_max = params.get('dist_transform_max', 10)

    # 1. Enhancement
    clahe = cv2.createCLAHE(clipLimit=0.05, tileGridSize=(3, 3))
    enhanced = clahe.apply(img_original.astype(np.uint8))

    # 2. Frangi Filter (Used for detection as requested)
    fr = frangi(
        enhanced,
        sigmas=np.arange(0.5, 4, 0.5),
        alpha=200,
        beta=1,
        gamma=10,
        black_ridges=False,
    )
    fr_norm = cv2.normalize(
        fr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
    ).astype(np.uint8)

    # 3. Skeletonization & Distance Transform
    fiber_core = (fr_norm > fiber_core_threshold).astype(np.uint8) * 255
    # We store the distance transform to calculate WIDTH later
    dist_transform = cv2.distanceTransform(fiber_core, cv2.DIST_L2, 3)

    skeleton = skeletonize(fiber_core > 0)
    skeleton_clean = (skeleton & (dist_transform <= dist_transform_max)).astype(np.uint8) * 255

    # Store skeleton before noise filtering for visualization
    skeleton_before_filter = skeleton_clean.copy()

    # 4. Filter small objects (noise filtering)
    labels = label(skeleton_clean)
    skeleton_connected = np.zeros_like(skeleton_clean, dtype=bool)
    for r in regionprops(labels):
        if r.area >= min_object_area:
            skeleton_connected[labels == r.label] = 1
    skeleton_connected = skeleton_connected.astype(np.uint8) * 255

    # Store skeleton after noise filtering for visualization
    skeleton_after_filter = skeleton_connected.copy()

    # 5. Blob identification
    _, blob_core = cv2.threshold(img_original, blob_threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    opened = cv2.morphologyEx(blob_core, cv2.MORPH_OPEN, kernel)
    opened = cv2.dilate(opened, kernel)
    inv_opened = cv2.bitwise_not(opened)

    # 6. Final Skeleton
    final_image = cv2.bitwise_and(skeleton_connected, inv_opened)
    skeleton_final = (skeletonize(final_image > 0).astype(np.uint8)) * 255

    # 7. Path Extraction & Junction Detection
    skel_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(
        (skeleton_final > 0).astype(np.uint8), -1, skel_kernel
    )
    endpoints = np.logical_and(skeleton_final > 0, neighbor_count == 1)

    # Junction points are where a skeleton pixel has more than 2 neighbors
    junction_points = np.logical_and(skeleton_final > 0, neighbor_count >= 3)

    h, w = skeleton_final.shape
    visited = np.zeros_like(skeleton_final, dtype=bool)
    fibers = []

    def get_neighbors(y, x):
        for dy, dx in [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skeleton_final[ny, nx] > 0:
                yield ny, nx

    def get_angle(p1, p2):
        """Calculate angle from p1 to p2 in radians."""
        dy = p2[0] - p1[0]
        dx = p2[1] - p1[1]
        return math.atan2(dy, dx)

    def angle_difference(a1, a2):
        """Calculate smallest angle difference between two angles."""
        diff = abs(a1 - a2)
        return min(diff, 2 * math.pi - diff)

    def select_best_neighbor(prev, curr, neighbors):
        """
        Select the neighbor that continues most straight from the incoming direction.
        At junctions, this picks the path with smallest angle change.
        """
        if prev is None or len(neighbors) == 1:
            return neighbors[0]

        # Calculate incoming direction (from prev to curr)
        incoming_angle = get_angle(prev, curr)
        # The "straight" continuation would be the same angle
        continuation_angle = incoming_angle

        best_neighbor = None
        min_angle_diff = float('inf')

        for nbr in neighbors:
            # Calculate outgoing angle (from curr to neighbor)
            outgoing_angle = get_angle(curr, nbr)
            # How much does this deviate from going straight?
            diff = angle_difference(continuation_angle, outgoing_angle)

            if diff < min_angle_diff:
                min_angle_diff = diff
                best_neighbor = nbr

        return best_neighbor

    for sy, sx in zip(*np.where(endpoints)):
        if visited[sy, sx]:
            continue
        path = [(sy, sx)]
        visited[sy, sx] = True
        curr, prev = (sy, sx), None
        while True:
            y, x = curr
            nbrs = [n for n in get_neighbors(y, x) if n != prev and not visited[n]]
            if not nbrs:
                break
            # Use angle-based selection at junctions instead of arbitrary choice
            nxt = select_best_neighbor(prev, curr, nbrs)
            path.append(nxt)
            visited[nxt] = True
            prev, curr = curr, nxt
        if len(path) > 2:
            fibers.append(path)

    # Return additional data for visualizations
    intermediate_images = {
        'skeleton_before_filter': skeleton_before_filter,
        'skeleton_after_filter': skeleton_after_filter,
        'frangi_output': fr_norm,
        'junction_points': junction_points,
        'endpoints': endpoints,
    }

    return skeleton_final, fibers, img_original, dist_transform, intermediate_images


# --- PDF GENERATION LOGIC ---
class FiberReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "Fiber Analysis Automated Report", ln=True, align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def create_pdf(image, df, summary, original_filename):
    pdf = FiberReport()
    pdf.add_page()

    # Header Info
    pdf.set_font("Arial", "I", 8)
    pdf.cell(0, 5, f"Source File: {original_filename}", ln=True, align="R")
    pdf.cell(
        0,
        5,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ln=True,
        align="R",
    )
    pdf.ln(5)

    # 1. Add Image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        cv2.imwrite(tmpfile.name, image)
        # Calculate aspect ratio to fit image properly
        img_h, img_w = image.shape[:2]
        display_w = 190
        display_h = (img_h / img_w) * display_w
        pdf.image(tmpfile.name, x=10, y=pdf.get_y(), w=display_w)
        pdf.set_y(pdf.get_y() + display_h + 10)  # Move cursor below image
        os.remove(tmpfile.name)

    # 2. Add Summary Stats
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Analysis Summary (Group Averages)", ln=True)
    pdf.set_font("Arial", "", 10)

    # Using formatting to ensure summary values are also consistent
    pdf.cell(0, 7, f"Total Fibers: {summary['Total Fibers']}", ln=True)
    pdf.cell(
        0, 7, f"Avg Length: {float(summary['Avg Length'].split()[0]):.3f} px", ln=True
    )
    pdf.cell(
        0, 7, f"Avg Width: {float(summary['Avg Width'].split()[0]):.3f} px", ln=True
    )
    pdf.cell(
        0, 7, f"Avg Curvature Rate: {float(summary['Avg Curvature Rate']):.3f}", ln=True
    )

    # --- PAGE BREAK BEFORE TABLE ---
    pdf.add_page()

    # 3. Add Table
    pdf.set_font("Arial", "B", 10)
    cols = ["ID", "Length (px)", "Width (px)", "Curvature Rate", "Aspect Ratio"]
    col_widths = [15, 35, 35, 35, 35]

    # Header (Now guaranteed to be at the top of a new page)
    for i, col in enumerate(cols):
        pdf.cell(col_widths[i], 10, col, border=1, align="C")
    pdf.ln()

    # Rows with 3-decimal rounding
    pdf.set_font("Arial", "", 9)
    for _, row in df.iterrows():
        if pdf.get_y() > 260:  # Threshold for footer safety
            pdf.add_page()
            # Re-draw headers on the new page if the table spans multiple pages
            pdf.set_font("Arial", "B", 10)
            for i, col in enumerate(cols):
                pdf.cell(col_widths[i], 10, col, border=1, align="C")
            pdf.ln()
            pdf.set_font("Arial", "", 9)

        pdf.cell(col_widths[0], 8, str(int(row["fiber_id"])), border=1, align="C")
        pdf.cell(col_widths[1], 8, f"{row['length_px']:.3f}", border=1, align="C")
        pdf.cell(col_widths[2], 8, f"{row['width_px']:.3f}", border=1, align="C")
        pdf.cell(col_widths[3], 8, f"{row['curvature_rate']:.3f}", border=1, align="C")
        pdf.cell(col_widths[4], 8, f"{row['aspect_ratio']:.3f}", border=1, align="C")
        pdf.ln()

    return pdf.output()


def analyze_fiber_stats(path, dist_map):
    """Calculates metrics for a single fiber path."""
    # 1. Length (Geodesic)
    length = sum(
        math.hypot(path[i][1] - path[i - 1][1], path[i][0] - path[i - 1][0])
        for i in range(1, len(path))
    )

    # 2. Width (Average)
    # The distance transform value at a skeleton pixel is the radius. Width = 2 * radius.
    widths = [dist_map[y, x] * 2 for y, x in path]
    avg_width = np.mean(widths)

    # 3. Curvature (Curvature Rate)
    # Curvature Rate = Geodesic Length / Euclidean Distance (End-to-End)
    euclidean_dist = math.hypot(path[-1][1] - path[0][1], path[-1][0] - path[0][0])
    curvature_rate = length / euclidean_dist if euclidean_dist > 0 else 1.0

    # 4. Aspect Ratio
    aspect_ratio = length / avg_width if avg_width > 0 else 0

    return {
        "length_px": round(length, 2),
        "width_px": round(avg_width, 2),
        "curvature_rate": round(curvature_rate, 3),
        "aspect_ratio": round(aspect_ratio, 2),
    }


# --- APP LAYOUT ---
st.title("🔬 Advanced Fiber Analysis")

# Sidebar with all configurable parameters
st.sidebar.header("🎛️ Analysis Settings")

st.sidebar.subheader("Fiber Length Filters")
min_length = st.sidebar.slider(
    "Min Fiber Length (px)",
    min_value=5, max_value=200, value=30,
    help="Minimum length to consider a fiber valid"
)
max_length = st.sidebar.slider(
    "Max Fiber Length (px)",
    min_value=50, max_value=2000, value=1000,
    help="Maximum length to consider a fiber valid (filters out artifacts)"
)

st.sidebar.subheader("Noise Filtering")
min_object_area = st.sidebar.slider(
    "Min Object Area (px)",
    min_value=1, max_value=100, value=10,
    help="Objects smaller than this area are filtered as noise"
)
fiber_core_threshold = st.sidebar.slider(
    "Fiber Detection Sensitivity",
    min_value=5, max_value=100, value=20,
    help="Lower = more sensitive (detects fainter fibers), Higher = stricter"
)

st.sidebar.subheader("Advanced Parameters")
blob_threshold = st.sidebar.slider(
    "Blob Threshold",
    min_value=5, max_value=100, value=21,
    help="Threshold for identifying and removing blob artifacts"
)
dist_transform_max = st.sidebar.slider(
    "Max Fiber Width Detection",
    min_value=5, max_value=50, value=10,
    help="Maximum distance transform value for skeleton cleaning"
)

# Visualization options
st.sidebar.subheader("📊 Display Options")
show_noise_filter_view = st.sidebar.checkbox("Show Noise Filtering Impact", value=False)
show_junction_overlay = st.sidebar.checkbox("Show Junction Points", value=True)
show_frangi_output = st.sidebar.checkbox("Show Frangi Filter Output", value=False)

uploaded_file = st.file_uploader("Upload fiber image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner("Analyzing fibers..."):
        start_time = time.time()

        # Prepare processing parameters
        processing_params = {
            'fiber_core_threshold': fiber_core_threshold,
            'min_object_area': min_object_area,
            'blob_threshold': blob_threshold,
            'dist_transform_max': dist_transform_max,
        }

        skeleton_final, fibers, img_original, dist_transform, intermediate_images = process_fiber_image(
            uploaded_file.read(), processing_params
        )

        # --- OPTIONAL: Frangi Filter Output View ---
        if show_frangi_output:
            st.subheader("🔍 Frangi Filter Output")
            frangi_colored = cv2.applyColorMap(intermediate_images['frangi_output'], cv2.COLORMAP_JET)
            st.image(frangi_colored, use_container_width=True, caption="Frangi filter response (fiber enhancement)")

        # --- OPTIONAL: Noise Filtering Impact View ---
        if show_noise_filter_view:
            st.subheader("🧹 Noise Filtering Impact")
            col_before, col_after = st.columns(2)
            with col_before:
                st.image(intermediate_images['skeleton_before_filter'],
                         use_container_width=True,
                         caption="Before Noise Filtering")
            with col_after:
                st.image(intermediate_images['skeleton_after_filter'],
                         use_container_width=True,
                         caption="After Noise Filtering")

        # Prepare Visualizations
        img_bgr = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)
        highlight_vis = img_bgr.copy()

        for path in fibers:
            length_px = sum(
                math.hypot(p2[1] - p1[1], p2[0] - p1[0])
                for p1, p2 in zip(path[:-1], path[1:])
            )

            # Apply both min and max length filters
            if length_px < min_length or length_px > max_length:
                continue

            # Highlight Logic (drawing the actual fiber)
            for y, x in path:
                cv2.circle(
                    highlight_vis, (x, y), 1, (255, 0, 255), -1
                )  # Magenta highlights

        # --- JUNCTION POINTS OVERLAY ---
        if show_junction_overlay:
            junction_points = intermediate_images['junction_points']
            endpoints = intermediate_images['endpoints']

            # Draw junction points in cyan (larger circles)
            for y, x in zip(*np.where(junction_points)):
                cv2.circle(highlight_vis, (x, y), 3, (255, 255, 0), -1)  # Cyan for junctions

            # Draw endpoints in yellow (smaller circles)
            for y, x in zip(*np.where(endpoints)):
                cv2.circle(highlight_vis, (x, y), 2, (0, 255, 255), -1)  # Yellow for endpoints

        # Display Results
        st.subheader("Highlighted Fiber View" + (" with Junction Points" if show_junction_overlay else ""))
        if show_junction_overlay:
            st.caption("🟡 Yellow = Endpoints | 🔵 Cyan = Junction Points | 🟣 Magenta = Fiber Paths")
        st.image(highlight_vis, use_container_width=True)

        # Data storage
        fiber_data = []
        img_bgr = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)
        analysis_vis = img_bgr.copy()

        # Process each detected fiber
        fiber_id_counter = 1
        for i, path in enumerate(fibers):
            stats = analyze_fiber_stats(path, dist_transform)
            # Apply both min and max length filters
            if stats["length_px"] < min_length or stats["length_px"] > max_length:
                continue
            stats["fiber_id"] = fiber_id_counter
            fiber_id_counter += 1
            fiber_data.append(stats)
            ys, xs = [p[0] for p in path], [p[1] for p in path]
            cv2.rectangle(
                analysis_vis,
                (min(xs) - 3, min(ys) - 3),
                (max(xs) + 3, max(ys) + 3),
                (0, 255, 0),
                1,
            )
            cv2.putText(
                analysis_vis,
                str(stats["fiber_id"]),
                (min(xs), min(ys) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
            )

        df = pd.DataFrame(fiber_data)

        # --- SUMMARY METRICS DISPLAY ON WEBSITE ---
        if not df.empty:
            st.subheader("📊 Analysis Summary")
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            with metric_col1:
                st.metric("Total Fibers", len(df))
            with metric_col2:
                st.metric("Avg Length", f"{df['length_px'].mean():.2f} px")
            with metric_col3:
                st.metric("Avg Width", f"{df['width_px'].mean():.2f} px")
            with metric_col4:
                st.metric("Avg Curvature Rate", f"{df['curvature_rate'].mean():.3f}")

            # Additional stats row
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            with stat_col1:
                st.metric("Min Length", f"{df['length_px'].min():.2f} px")
            with stat_col2:
                st.metric("Max Length", f"{df['length_px'].max():.2f} px")
            with stat_col3:
                st.metric("Avg Aspect Ratio", f"{df['aspect_ratio'].mean():.2f}")
            with stat_col4:
                junction_count = np.sum(intermediate_images['junction_points'])
                st.metric("Junction Points", junction_count)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(
                analysis_vis,
                use_container_width=True,
                caption="Detection Visualization",
            )
        with col2:
            if not df.empty:
                summary = {
                    "Total Fibers": len(df),
                    "Avg Length": f"{df['length_px'].mean():.2f} px",
                    "Avg Width": f"{df['width_px'].mean():.2f} px",
                    "Avg Curvature Rate": f"{df['curvature_rate'].mean():.3f}",
                }
                st.subheader("Detailed Metrics")
                st.dataframe(
                    df[
                        [
                            "fiber_id",
                            "length_px",
                            "width_px",
                            "curvature_rate",
                            "aspect_ratio",
                        ]
                    ],
                    use_container_width=True,
                )

            else:
                st.warning("No fibers detected.")

        # --- METRIC VISUALIZATIONS ---
        if not df.empty:
            st.subheader("📈 Metric Visualizations")

            # Create tabs for different visualization types
            viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Length & Width", "Curvature Analysis", "Distribution Overview"])

            with viz_tab1:
                viz_col1, viz_col2 = st.columns(2)

                with viz_col1:
                    # Histogram for fiber lengths
                    fig_length = px.histogram(
                        df, x='length_px',
                        nbins=20,
                        title='Fiber Length Distribution',
                        labels={'length_px': 'Length (px)', 'count': 'Count'},
                        color_discrete_sequence=['#FF6B6B']
                    )
                    fig_length.update_layout(
                        showlegend=False,
                        xaxis_title="Length (px)",
                        yaxis_title="Count"
                    )
                    st.plotly_chart(fig_length, use_container_width=True)

                with viz_col2:
                    # Histogram for fiber widths
                    fig_width = px.histogram(
                        df, x='width_px',
                        nbins=20,
                        title='Fiber Width Distribution',
                        labels={'width_px': 'Width (px)', 'count': 'Count'},
                        color_discrete_sequence=['#4ECDC4']
                    )
                    fig_width.update_layout(
                        showlegend=False,
                        xaxis_title="Width (px)",
                        yaxis_title="Count"
                    )
                    st.plotly_chart(fig_width, use_container_width=True)

            with viz_tab2:
                viz_col3, viz_col4 = st.columns(2)

                with viz_col3:
                    # Curvature rate distribution (box plot)
                    fig_curv = px.box(
                        df, y='curvature_rate',
                        title='Curvature Rate Distribution',
                        labels={'curvature_rate': 'Curvature Rate'},
                        color_discrete_sequence=['#45B7D1']
                    )
                    fig_curv.update_layout(showlegend=False)
                    st.plotly_chart(fig_curv, use_container_width=True)

                with viz_col4:
                    # Scatter plot: Length vs Curvature Rate
                    fig_scatter = px.scatter(
                        df, x='length_px', y='curvature_rate',
                        title='Length vs Curvature Rate',
                        labels={'length_px': 'Length (px)', 'curvature_rate': 'Curvature Rate'},
                        color='width_px',
                        color_continuous_scale='Viridis',
                        hover_data=['fiber_id', 'aspect_ratio']
                    )
                    fig_scatter.update_layout(coloraxis_colorbar_title="Width (px)")
                    st.plotly_chart(fig_scatter, use_container_width=True)

            with viz_tab3:
                # Comprehensive overview with subplots
                fig_overview = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Length Distribution', 'Width Distribution',
                                    'Aspect Ratio Distribution', 'Curvature Rate Histogram')
                )

                # Length histogram
                fig_overview.add_trace(
                    go.Histogram(x=df['length_px'], name='Length', marker_color='#FF6B6B', nbinsx=15),
                    row=1, col=1
                )

                # Width histogram
                fig_overview.add_trace(
                    go.Histogram(x=df['width_px'], name='Width', marker_color='#4ECDC4', nbinsx=15),
                    row=1, col=2
                )

                # Aspect ratio histogram
                fig_overview.add_trace(
                    go.Histogram(x=df['aspect_ratio'], name='Aspect Ratio', marker_color='#96CEB4', nbinsx=15),
                    row=2, col=1
                )

                # Curvature rate histogram
                fig_overview.add_trace(
                    go.Histogram(x=df['curvature_rate'], name='Curvature Rate', marker_color='#45B7D1', nbinsx=15),
                    row=2, col=2
                )

                fig_overview.update_layout(
                    height=600,
                    showlegend=False,
                    title_text="Fiber Metrics Overview"
                )
                st.plotly_chart(fig_overview, use_container_width=True)

        if not df.empty:
            # Generate PDF
            raw_pdf = create_pdf(analysis_vis, df, summary, uploaded_file.name)
            pdf_bytes = bytes(raw_pdf)  # The crucial conversion

            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name=f"Fiber_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
            )

        st.success(f"Processing complete in {time.time() - start_time:.2f}s")
else:
    st.info("Upload an image to begin analysis.")
