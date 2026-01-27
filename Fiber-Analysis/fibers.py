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

def get_skeleton_and_junctions(img_original, params):
    """
    Common preprocessing: returns skeleton, junctions, endpoints, and other shared data.
    """
    fiber_core_threshold = params.get('fiber_core_threshold', 20)
    min_object_area = params.get('min_object_area', 10)
    blob_threshold = params.get('blob_threshold', 21)
    dist_transform_max = params.get('dist_transform_max', 10)

    # 1. Enhancement
    clahe = cv2.createCLAHE(clipLimit=0.05, tileGridSize=(3, 3))
    enhanced = clahe.apply(img_original.astype(np.uint8))

    # 2. Frangi Filter
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
    dist_transform = cv2.distanceTransform(fiber_core, cv2.DIST_L2, 3)

    skeleton = skeletonize(fiber_core > 0)
    skeleton_clean = (skeleton & (dist_transform <= dist_transform_max)).astype(np.uint8) * 255

    skeleton_before_filter = skeleton_clean.copy()

    # 4. Filter small objects
    labels = label(skeleton_clean)
    skeleton_connected = np.zeros_like(skeleton_clean, dtype=bool)
    for r in regionprops(labels):
        if r.area >= min_object_area:
            skeleton_connected[labels == r.label] = 1
    skeleton_connected = skeleton_connected.astype(np.uint8) * 255

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

    # 7. Junction Detection
    skel_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(
        (skeleton_final > 0).astype(np.uint8), -1, skel_kernel
    )
    endpoints = np.logical_and(skeleton_final > 0, neighbor_count == 1)
    junction_points = np.logical_and(skeleton_final > 0, neighbor_count >= 3)

    return {
        'skeleton_final': skeleton_final,
        'dist_transform': dist_transform,
        'endpoints': endpoints,
        'junction_points': junction_points,
        'neighbor_count': neighbor_count,
        'skeleton_before_filter': skeleton_before_filter,
        'skeleton_after_filter': skeleton_after_filter,
        'frangi_output': fr_norm,
    }


def get_path_intensity(path, img, window=5):
    """Get average intensity of last 'window' pixels in path."""
    if len(path) < window:
        points = path
    else:
        points = path[-window:]
    intensities = [img[y, x] for y, x in points]
    return np.mean(intensities)


def get_neighbor_intensity(curr, neighbor, img, skeleton_final, lookahead=3):
    """Look ahead from neighbor and get average intensity."""
    h, w = skeleton_final.shape
    intensities = [img[neighbor[0], neighbor[1]]]

    def get_neighbors(y, x):
        for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skeleton_final[ny, nx] > 0:
                yield ny, nx

    prev, current = curr, neighbor
    for _ in range(lookahead - 1):
        nbrs = [n for n in get_neighbors(current[0], current[1]) if n != prev]
        if not nbrs:
            break
        nxt = nbrs[0]  # Just follow any path for lookahead
        intensities.append(img[nxt[0], nxt[1]])
        prev, current = current, nxt

    return np.mean(intensities)


# === METHOD 1: ANGLE-ONLY (Original) ===
def extract_fibers_angle_only(skeleton_data, img_original):
    """Original method: angle-based path continuation only."""
    skeleton_final = skeleton_data['skeleton_final']
    endpoints = skeleton_data['endpoints']
    h, w = skeleton_final.shape
    visited = np.zeros_like(skeleton_final, dtype=bool)
    fibers = []

    def get_neighbors(y, x):
        for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skeleton_final[ny, nx] > 0:
                yield ny, nx

    def get_angle(p1, p2):
        dy = p2[0] - p1[0]
        dx = p2[1] - p1[1]
        return math.atan2(dy, dx)

    def angle_difference(a1, a2):
        diff = abs(a1 - a2)
        return min(diff, 2 * math.pi - diff)

    def select_best_neighbor(prev, curr, neighbors):
        if prev is None or len(neighbors) == 1:
            return neighbors[0]
        incoming_angle = get_angle(prev, curr)
        best_neighbor = None
        min_angle_diff = float('inf')
        for nbr in neighbors:
            outgoing_angle = get_angle(curr, nbr)
            diff = angle_difference(incoming_angle, outgoing_angle)
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
            nxt = select_best_neighbor(prev, curr, nbrs)
            path.append(nxt)
            visited[nxt] = True
            prev, curr = curr, nxt
        if len(path) > 2:
            fibers.append(path)

    return fibers


# === METHOD 2: INTENSITY-WEIGHTED JUNCTION DECISIONS ===
def extract_fibers_intensity_weighted(skeleton_data, img_original, intensity_weight=0.5):
    """
    At junctions, combine angle score with intensity similarity.
    Favors paths where intensity matches the incoming fiber's intensity profile.
    """
    skeleton_final = skeleton_data['skeleton_final']
    endpoints = skeleton_data['endpoints']
    h, w = skeleton_final.shape
    visited = np.zeros_like(skeleton_final, dtype=bool)
    fibers = []

    def get_neighbors(y, x):
        for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skeleton_final[ny, nx] > 0:
                yield ny, nx

    def get_angle(p1, p2):
        dy = p2[0] - p1[0]
        dx = p2[1] - p1[1]
        return math.atan2(dy, dx)

    def angle_difference(a1, a2):
        diff = abs(a1 - a2)
        return min(diff, 2 * math.pi - diff)

    def select_best_neighbor_weighted(prev, curr, neighbors, path):
        if prev is None or len(neighbors) == 1:
            return neighbors[0]

        incoming_angle = get_angle(prev, curr)
        path_intensity = get_path_intensity(path, img_original, window=5)

        best_neighbor = None
        best_score = float('inf')

        for nbr in neighbors:
            # Angle score (0 = straight, pi = U-turn)
            outgoing_angle = get_angle(curr, nbr)
            angle_diff = angle_difference(incoming_angle, outgoing_angle)
            angle_score = angle_diff / math.pi  # Normalize to 0-1

            # Intensity score (difference from path intensity)
            nbr_intensity = get_neighbor_intensity(curr, nbr, img_original, skeleton_final, lookahead=3)
            intensity_diff = abs(path_intensity - nbr_intensity) / 255.0  # Normalize to 0-1

            # Combined score (lower is better)
            combined_score = (1 - intensity_weight) * angle_score + intensity_weight * intensity_diff

            if combined_score < best_score:
                best_score = combined_score
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
            nxt = select_best_neighbor_weighted(prev, curr, nbrs, path)
            path.append(nxt)
            visited[nxt] = True
            prev, curr = curr, nxt
        if len(path) > 2:
            fibers.append(path)

    return fibers


# === METHOD 3: POST-PROCESSING FIBER MERGING ===
def extract_fibers_with_merging(skeleton_data, img_original, intensity_threshold=25, angle_threshold=45):
    """
    Extract fibers with angle-only, then merge segments at junctions
    if they have similar intensities and compatible directions.
    """
    skeleton_final = skeleton_data['skeleton_final']
    junction_points = skeleton_data['junction_points']

    # First, get fibers using angle-only method
    fibers = extract_fibers_angle_only(skeleton_data, img_original)

    if len(fibers) < 2:
        return fibers

    # Calculate intensity profile for each fiber
    fiber_intensities = []
    for path in fibers:
        intensities = [img_original[y, x] for y, x in path]
        fiber_intensities.append({
            'mean': np.mean(intensities),
            'std': np.std(intensities),
            'start': path[0],
            'end': path[-1],
            'start_angle': math.atan2(path[1][0] - path[0][0], path[1][1] - path[0][1]) if len(path) > 1 else 0,
            'end_angle': math.atan2(path[-1][0] - path[-2][0], path[-1][1] - path[-2][1]) if len(path) > 1 else 0,
        })

    # Find junction locations
    junction_coords = set(zip(*np.where(junction_points)))

    def is_near_junction(point, radius=3):
        y, x = point
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if (y + dy, x + dx) in junction_coords:
                    return True
        return False

    def points_close(p1, p2, threshold=5):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1]) <= threshold

    def angles_compatible(a1, a2, threshold_deg):
        # Check if angles are roughly opposite (continuation)
        diff = abs(a1 - a2)
        diff = min(diff, 2 * math.pi - diff)
        # For continuation, angles should be ~180 degrees apart
        continuation_diff = abs(diff - math.pi)
        return continuation_diff < math.radians(threshold_deg)

    # Try to merge fibers
    merged = [False] * len(fibers)
    merged_fibers = []

    for i in range(len(fibers)):
        if merged[i]:
            continue

        current_fiber = list(fibers[i])
        current_info = fiber_intensities[i]
        changed = True

        while changed:
            changed = False
            for j in range(len(fibers)):
                if i == j or merged[j]:
                    continue

                other_info = fiber_intensities[j]
                intensity_diff = abs(current_info['mean'] - other_info['mean'])

                if intensity_diff > intensity_threshold:
                    continue

                # Check if endpoints meet at a junction
                # Current end -> Other start
                if (is_near_junction(current_fiber[-1]) and
                    points_close(current_fiber[-1], fibers[j][0]) and
                    angles_compatible(current_info['end_angle'], other_info['start_angle'], angle_threshold)):
                    current_fiber.extend(fibers[j][1:])
                    current_info['end'] = other_info['end']
                    current_info['end_angle'] = other_info['end_angle']
                    current_info['mean'] = np.mean([img_original[y, x] for y, x in current_fiber])
                    merged[j] = True
                    changed = True
                    break

                # Current end -> Other end (reverse other)
                if (is_near_junction(current_fiber[-1]) and
                    points_close(current_fiber[-1], fibers[j][-1]) and
                    angles_compatible(current_info['end_angle'], -other_info['end_angle'], angle_threshold)):
                    current_fiber.extend(reversed(fibers[j][:-1]))
                    current_info['end'] = other_info['start']
                    current_info['end_angle'] = -other_info['start_angle']
                    current_info['mean'] = np.mean([img_original[y, x] for y, x in current_fiber])
                    merged[j] = True
                    changed = True
                    break

                # Current start -> Other end
                if (is_near_junction(current_fiber[0]) and
                    points_close(current_fiber[0], fibers[j][-1]) and
                    angles_compatible(-current_info['start_angle'], other_info['end_angle'], angle_threshold)):
                    current_fiber = list(fibers[j]) + current_fiber[1:]
                    current_info['start'] = other_info['start']
                    current_info['start_angle'] = other_info['start_angle']
                    current_info['mean'] = np.mean([img_original[y, x] for y, x in current_fiber])
                    merged[j] = True
                    changed = True
                    break

                # Current start -> Other start (reverse other)
                if (is_near_junction(current_fiber[0]) and
                    points_close(current_fiber[0], fibers[j][0]) and
                    angles_compatible(-current_info['start_angle'], -other_info['start_angle'], angle_threshold)):
                    current_fiber = list(reversed(fibers[j])) + current_fiber[1:]
                    current_info['start'] = other_info['end']
                    current_info['start_angle'] = -other_info['end_angle']
                    current_info['mean'] = np.mean([img_original[y, x] for y, x in current_fiber])
                    merged[j] = True
                    changed = True
                    break

        merged_fibers.append(current_fiber)
        merged[i] = True

    return merged_fibers


# === METHOD 4: INTENSITY CORRIDOR TRACKING ===
def extract_fibers_intensity_corridor(skeleton_data, img_original, max_intensity_deviation=30):
    """
    Track along an intensity corridor - penalize paths that deviate
    too much from the running average intensity.
    """
    skeleton_final = skeleton_data['skeleton_final']
    endpoints = skeleton_data['endpoints']
    h, w = skeleton_final.shape
    visited = np.zeros_like(skeleton_final, dtype=bool)
    fibers = []

    def get_neighbors(y, x):
        for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skeleton_final[ny, nx] > 0:
                yield ny, nx

    def get_angle(p1, p2):
        dy = p2[0] - p1[0]
        dx = p2[1] - p1[1]
        return math.atan2(dy, dx)

    def angle_difference(a1, a2):
        diff = abs(a1 - a2)
        return min(diff, 2 * math.pi - diff)

    for sy, sx in zip(*np.where(endpoints)):
        if visited[sy, sx]:
            continue
        path = [(sy, sx)]
        visited[sy, sx] = True
        curr, prev = (sy, sx), None

        # Running intensity estimate (exponential moving average)
        running_intensity = float(img_original[sy, sx])
        alpha = 0.3  # Smoothing factor

        while True:
            y, x = curr
            nbrs = [n for n in get_neighbors(y, x) if n != prev and not visited[n]]
            if not nbrs:
                break

            if prev is None or len(nbrs) == 1:
                nxt = nbrs[0]
            else:
                # Score each neighbor
                incoming_angle = get_angle(prev, curr)
                best_neighbor = None
                best_score = float('inf')

                for nbr in nbrs:
                    # Angle component
                    outgoing_angle = get_angle(curr, nbr)
                    angle_diff = angle_difference(incoming_angle, outgoing_angle)

                    # Intensity component - how much does this pixel deviate?
                    nbr_intensity = float(img_original[nbr[0], nbr[1]])
                    intensity_deviation = abs(nbr_intensity - running_intensity)

                    # If deviation is too large, heavily penalize
                    if intensity_deviation > max_intensity_deviation:
                        intensity_penalty = 2.0 + (intensity_deviation - max_intensity_deviation) / 50.0
                    else:
                        intensity_penalty = intensity_deviation / max_intensity_deviation

                    # Combined score
                    score = angle_diff + intensity_penalty * 0.5

                    if score < best_score:
                        best_score = score
                        best_neighbor = nbr

                nxt = best_neighbor

            # Update running intensity
            nxt_intensity = float(img_original[nxt[0], nxt[1]])
            running_intensity = alpha * nxt_intensity + (1 - alpha) * running_intensity

            path.append(nxt)
            visited[nxt] = True
            prev, curr = curr, nxt

        if len(path) > 2:
            fibers.append(path)

    return fibers


def process_fiber_image(image_bytes, params, method='angle_only'):
    """
    Process fiber image with configurable parameters and method.

    Methods:
        - 'angle_only': Original angle-based method
        - 'intensity_weighted': Intensity-weighted junction decisions
        - 'post_merge': Post-processing fiber merging
        - 'intensity_corridor': Intensity corridor tracking
    """
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_original = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # Get skeleton and junction data
    skeleton_data = get_skeleton_and_junctions(img_original, params)

    # Extract fibers using selected method
    intensity_weight = params.get('intensity_weight', 0.5)
    intensity_threshold = params.get('intensity_threshold', 25)
    angle_threshold = params.get('merge_angle_threshold', 45)
    max_intensity_deviation = params.get('max_intensity_deviation', 30)

    if method == 'angle_only':
        fibers = extract_fibers_angle_only(skeleton_data, img_original)
    elif method == 'intensity_weighted':
        fibers = extract_fibers_intensity_weighted(skeleton_data, img_original, intensity_weight)
    elif method == 'post_merge':
        fibers = extract_fibers_with_merging(skeleton_data, img_original, intensity_threshold, angle_threshold)
    elif method == 'intensity_corridor':
        fibers = extract_fibers_intensity_corridor(skeleton_data, img_original, max_intensity_deviation)
    else:
        fibers = extract_fibers_angle_only(skeleton_data, img_original)

    # Return additional data for visualizations
    intermediate_images = {
        'skeleton_before_filter': skeleton_data['skeleton_before_filter'],
        'skeleton_after_filter': skeleton_data['skeleton_after_filter'],
        'frangi_output': skeleton_data['frangi_output'],
        'junction_points': skeleton_data['junction_points'],
        'endpoints': skeleton_data['endpoints'],
    }

    return skeleton_data['skeleton_final'], fibers, img_original, skeleton_data['dist_transform'], intermediate_images


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
    min_value=5, max_value=200, value=15,
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
    min_value=5, max_value=100, value=30,
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

# Intensity-based method parameters
st.sidebar.subheader("Intensity-Based Methods")
intensity_weight = st.sidebar.slider(
    "Intensity Weight (Method 2)",
    min_value=0.0, max_value=1.0, value=0.5, step=0.1,
    help="Weight for intensity vs angle in junction decisions (0=angle only, 1=intensity only)"
)
intensity_threshold = st.sidebar.slider(
    "Merge Intensity Threshold (Method 3)",
    min_value=5, max_value=100, value=25,
    help="Max intensity difference for merging fibers"
)
merge_angle_threshold = st.sidebar.slider(
    "Merge Angle Threshold (Method 3)",
    min_value=15, max_value=90, value=45,
    help="Max angle deviation (degrees) for merging fibers"
)
max_intensity_deviation = st.sidebar.slider(
    "Max Intensity Deviation (Method 4)",
    min_value=10, max_value=100, value=30,
    help="Maximum allowed intensity deviation from running average"
)

# Method selection for main analysis
st.sidebar.subheader("🎯 Analysis Method")
selected_method = st.sidebar.selectbox(
    "Method for Report",
    options=['angle_only', 'intensity_weighted', 'post_merge', 'intensity_corridor'],
    format_func=lambda x: {
        'angle_only': '1: Angle Only (Original)',
        'intensity_weighted': '2: Intensity-Weighted Junctions',
        'post_merge': '3: Post-Processing Merge',
        'intensity_corridor': '4: Intensity Corridor',
    }[x],
    help="Select which method to use for the main analysis and PDF report"
)

# Visualization options
st.sidebar.subheader("📊 Display Options")
show_noise_filter_view = st.sidebar.checkbox("Show Noise Filtering Impact", value=True)
show_junction_overlay = st.sidebar.checkbox("Show Junction Points", value=True)
show_frangi_output = st.sidebar.checkbox("Show Frangi Filter Output", value=False)
show_method_comparison = st.sidebar.checkbox("Show Method Comparison", value=True)

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
            'intensity_weight': intensity_weight,
            'intensity_threshold': intensity_threshold,
            'merge_angle_threshold': merge_angle_threshold,
            'max_intensity_deviation': max_intensity_deviation,
        }

        # Read the file once
        file_bytes = uploaded_file.read()

        # Process with selected method for main analysis
        skeleton_final, fibers, img_original, dist_transform, intermediate_images = process_fiber_image(
            file_bytes, processing_params, method=selected_method
        )

        # --- OPTIONAL: Frangi Filter Output View ---
        if show_frangi_output:
            st.subheader("🔍 Frangi Filter Output")
            frangi_colored = cv2.applyColorMap(intermediate_images['frangi_output'], cv2.COLORMAP_JET)
            st.image(frangi_colored, use_container_width=True, caption="Frangi filter response (fiber enhancement)")

        # --- METHOD COMPARISON VIEW ---
        if show_method_comparison:
            st.subheader("🔬 Intensity Method Comparison")
            st.caption("Compare all 4 methods to see which produces the best fiber coherence for your images")

            # Process with all methods
            methods = {
                'angle_only': 'Method 1: Angle Only (Original)',
                'intensity_weighted': 'Method 2: Intensity-Weighted Junctions',
                'post_merge': 'Method 3: Post-Processing Merge',
                'intensity_corridor': 'Method 4: Intensity Corridor',
            }

            method_fibers = {}
            method_stats = {}

            for method_key in methods.keys():
                _, method_fiber_list, _, _, _ = process_fiber_image(
                    file_bytes, processing_params, method=method_key
                )
                method_fibers[method_key] = method_fiber_list

                # Count valid fibers (applying length filters)
                valid_count = 0
                total_length = 0
                for path in method_fiber_list:
                    length_px = sum(
                        math.hypot(p2[1] - p1[1], p2[0] - p1[0])
                        for p1, p2 in zip(path[:-1], path[1:])
                    )
                    if min_length <= length_px <= max_length:
                        valid_count += 1
                        total_length += length_px

                method_stats[method_key] = {
                    'count': valid_count,
                    'avg_length': total_length / valid_count if valid_count > 0 else 0
                }

            # Create comparison visualizations
            # Use distinct colors for each fiber
            def create_method_visualization(method_key, img_base):
                vis = cv2.cvtColor(img_base.copy(), cv2.COLOR_GRAY2BGR)
                fiber_list = method_fibers[method_key]

                # Generate distinct colors for fibers
                np.random.seed(42)  # Consistent colors across methods
                colors = []
                for i in range(len(fiber_list)):
                    hue = (i * 137) % 360  # Golden angle for good distribution
                    # Convert HSV to BGR
                    h = hue / 2  # OpenCV uses 0-180 for hue
                    color_hsv = np.uint8([[[h, 255, 255]]])
                    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
                    colors.append(tuple(map(int, color_bgr)))

                fiber_idx = 0
                for path in fiber_list:
                    length_px = sum(
                        math.hypot(p2[1] - p1[1], p2[0] - p1[0])
                        for p1, p2 in zip(path[:-1], path[1:])
                    )
                    if length_px < min_length or length_px > max_length:
                        continue

                    color = colors[fiber_idx % len(colors)] if colors else (255, 0, 255)
                    for y, x in path:
                        cv2.circle(vis, (x, y), 1, color, -1)
                    fiber_idx += 1

                return vis

            # Display in 2x2 grid
            col1, col2 = st.columns(2)

            with col1:
                vis1 = create_method_visualization('angle_only', img_original)
                st.image(vis1, use_container_width=True,
                         caption=f"Method 1: Angle Only - {method_stats['angle_only']['count']} fibers, avg {method_stats['angle_only']['avg_length']:.1f}px")

                vis3 = create_method_visualization('post_merge', img_original)
                st.image(vis3, use_container_width=True,
                         caption=f"Method 3: Post-Merge - {method_stats['post_merge']['count']} fibers, avg {method_stats['post_merge']['avg_length']:.1f}px")

            with col2:
                vis2 = create_method_visualization('intensity_weighted', img_original)
                st.image(vis2, use_container_width=True,
                         caption=f"Method 2: Intensity-Weighted - {method_stats['intensity_weighted']['count']} fibers, avg {method_stats['intensity_weighted']['avg_length']:.1f}px")

                vis4 = create_method_visualization('intensity_corridor', img_original)
                st.image(vis4, use_container_width=True,
                         caption=f"Method 4: Intensity Corridor - {method_stats['intensity_corridor']['count']} fibers, avg {method_stats['intensity_corridor']['avg_length']:.1f}px")

            # Summary table
            st.markdown("### Method Comparison Summary")
            comparison_df = pd.DataFrame({
                'Method': [methods[k] for k in methods.keys()],
                'Fiber Count': [method_stats[k]['count'] for k in methods.keys()],
                'Avg Length (px)': [f"{method_stats[k]['avg_length']:.1f}" for k in methods.keys()],
            })
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

            st.info("""
            **Interpretation Guide:**
            - **Fewer fibers with longer average length** = better merging of coherent fiber segments
            - **Method 1 (Angle Only)**: Baseline - uses only geometric continuity
            - **Method 2 (Intensity-Weighted)**: At junctions, prefers paths with similar brightness
            - **Method 3 (Post-Merge)**: Merges segments after extraction if intensities match
            - **Method 4 (Intensity Corridor)**: Tracks along consistent brightness paths

            Each fiber is shown in a different color. Look for cases where the same physical fiber is split into multiple colors (bad) vs shown as one color (good).
            """)

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
