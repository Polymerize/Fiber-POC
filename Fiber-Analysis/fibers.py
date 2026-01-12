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
def process_fiber_image(image_bytes):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_original = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

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
    fiber_core = (fr_norm > 20).astype(np.uint8) * 255
    # We store the distance transform to calculate WIDTH later
    dist_transform = cv2.distanceTransform(fiber_core, cv2.DIST_L2, 3)

    skeleton = skeletonize(fiber_core > 0)
    skeleton_clean = (skeleton & (dist_transform <= 10)).astype(np.uint8) * 255

    # 4. Filter small objects
    labels = label(skeleton_clean)
    skeleton_connected = np.zeros_like(skeleton_clean, dtype=bool)
    for r in regionprops(labels):
        if r.area >= 10:
            skeleton_connected[labels == r.label] = 1
    skeleton_connected = skeleton_connected.astype(np.uint8) * 255

    # 5. Blob identification
    blob_threshold = 21
    _, blob_core = cv2.threshold(img_original, blob_threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    opened = cv2.morphologyEx(blob_core, cv2.MORPH_OPEN, kernel)
    opened = cv2.dilate(opened, kernel)
    inv_opened = cv2.bitwise_not(opened)

    # 6. Final Skeleton
    final_image = cv2.bitwise_and(skeleton_connected, inv_opened)
    skeleton_final = (skeletonize(final_image > 0).astype(np.uint8)) * 255

    # 7. Path Extraction (Your original logic)
    skel_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(
        (skeleton_final > 0).astype(np.uint8), -1, skel_kernel
    )
    endpoints = np.logical_and(skeleton_final > 0, neighbor_count == 1)

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
            nxt = nbrs[0]
            path.append(nxt)
            visited[nxt] = True
            prev, curr = curr, nxt
        if len(path) > 2:
            fibers.append(path)

    return skeleton_final, fibers, img_original, dist_transform


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
st.sidebar.header("Analysis Settings")
min_length = st.sidebar.slider("Min Fiber Length (px)", 5, 200, 30)

uploaded_file = st.file_uploader("Upload fiber image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner("Analyzing fibers..."):
        start_time = time.time()
        skeleton_final, fibers, img_original, dist_transform = process_fiber_image(
            uploaded_file.read()
        )
        # Prepare Visualizations
        img_bgr = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)
        highlight_vis = img_bgr.copy()

        for path in fibers:
            length_px = sum(
                math.hypot(p2[1] - p1[1], p2[0] - p1[0])
                for p1, p2 in zip(path[:-1], path[1:])
            )

            if length_px < min_length:
                continue

            # Bounding Box Logic (not used here)
            # color = (0, 255, 0) # Green

            # ys, xs = [p[0] for p in path], [p[1] for p in path]
            # x1, y1, x2, y2 = min(xs)-3, min(ys)-3, max(xs)+3, max(ys)+3
            # cv2.rectangle(bbox_vis, (x1, y1), (x2, y2), color, 1)
            # cv2.putText(bbox_vis, f"{length_px:.1f}", (x1, max(y1-5, 10)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Highlight Logic (drawing the actual fiber)
            for y, x in path:
                cv2.circle(
                    highlight_vis, (x, y), 1, (255, 0, 255), -1
                )  # Magenta highlights

        # changed since bounding box view is separately used after this step
        # col1, col2 = st.columns(2)
        # with col1:
        #     st.subheader("Bounding Box View")
        #     st.image(bbox_vis, use_container_width=True)

        # with col2:

        # Display Results
        st.subheader("Highlighted Fiber View")
        st.image(highlight_vis, use_container_width=True)

        # Data storage
        fiber_data = []
        img_bgr = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)
        analysis_vis = img_bgr.copy()

        # Process each detected fiber
        for i, path in enumerate(fibers):
            stats = analyze_fiber_stats(path, dist_transform)
            if stats["length_px"] < min_length:
                continue
            stats["fiber_id"] = i + 1
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
                str(i + 1),
                (min(xs), min(ys) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
            )

        df = pd.DataFrame(fiber_data)

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
