
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("Fiber Detection and Analysis App")
st.write("Upload an image to detect and analyze fibers.")

# --- Streamlit UI for Image Upload ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an opencv image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    st.subheader("Uploaded Image")
    st.image(original_image, caption="Original Uploaded Image (Grayscale)", use_container_width=True, channels="GRAY")
    
    st.session_state['original_image'] = original_image
    st.success("Image uploaded successfully!")
else:
    st.warning("Please upload an image to proceed.")


if 'original_image' in st.session_state and st.session_state['original_image'] is not None:
    original_image = st.session_state['original_image']

    st.subheader("Step 1: Background Masking")
    background_value = st.slider(
        'Background Removal Sensitivity (high = more pixels will be removed)',
        0, 255, 50, # Min, Max, Default value
        key='background_slider'
    )

    # Apply binary thresholding
    _, background_removed_mask = cv2.threshold(original_image, background_value, 255, cv2.THRESH_BINARY)

    st.image(background_removed_mask, caption=f'Background Removed Mask (Threshold: {background_value})', use_container_width=True, channels="GRAY")

    st.session_state['background_removed_mask'] = background_removed_mask
    st.session_state['background_value'] = background_value

print("Added background masking logic with sensitivity slider to main.py")

if 'background_removed_mask' in st.session_state and st.session_state['background_removed_mask'] is not None:
    original_image = st.session_state['original_image']
    background_removed_mask = st.session_state['background_removed_mask']

    st.subheader("Step 2: Fiber Detection")

    min_fiber_area = st.slider(
        'Minimum Fiber Area (pixels) to skip detecting noise from scans',
        0, 200, 5, # Min, Max, Default value
        key='min_fiber_area_slider'
    )
    
    fiber_run = st.checkbox('Proceed to Fiber Detection')
    if fiber_run:

        # Connected components analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(background_removed_mask, 8, cv2.CV_32S)

        # Prepare an output image for drawing (BGR format)
        output_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

        detected_count = 0
        for i in range(1, num_labels): # Skip background component (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area > min_fiber_area:
                detected_count += 1

                # Highlight the component pixels in green
                output_image[labels == i] = (0, 255, 0) # Green color

                # Get the bounding box coordinates
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]

                # Draw a bounding box around the highlighted component
                cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 0, 255), 1) # Red bounding box, thickness 1

        st.image(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), caption=f'Detected Fibers (Count: {detected_count}, Min Area: {min_fiber_area})', use_container_width=True)

        st.session_state['cleaned_mask'] = background_removed_mask
        st.session_state['min_fiber_area'] = min_fiber_area
        st.session_state['num_labels'] = num_labels
        st.session_state['labels'] = labels
        st.session_state['stats'] = stats
        st.session_state['centroids'] = centroids
        st.session_state['output_image'] = output_image # Store the image with highlights and bboxes

        st.success(f"Fiber detection complete. Found {detected_count} fibers.")
    else:
        st.info("Click 'Proceed to Fiber Detection' to analyze fibers.")

# Define the bounding box utility functions outside the main conditional blocks
# so they are only defined once when the script loads.
import math

def do_bboxes_overlap(bbox1, bbox2):
    """Checks if two bounding boxes overlap."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Check for x-axis overlap
    x_overlap = (x1 < x2 + w2) and (x2 < x1 + w1)
    # Check for y-axis overlap
    y_overlap = (y1 < y2 + h2) and (y2 < y1 + h1)

    return x_overlap and y_overlap

def are_bboxes_close(bbox1, bbox2, max_distance):
    """Checks if two non-overlapping bounding boxes are close enough based on max_distance."""
    if do_bboxes_overlap(bbox1, bbox2):
        return False # This function is for non-overlapping boxes

    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Calculate the coordinates of the corners for easier comparison
    left1, top1, right1, bottom1 = x1, y1, x1 + w1, y1 + h1
    left2, top2, right2, bottom2 = x2, y2, x2 + w2, y2 + h2

    # Calculate horizontal and vertical distances between the boxes
    # If intervals overlap, distance is 0 in that dimension, otherwise it's the gap.
    dx = max(0, left1 - right2, left2 - right1)
    dy = max(0, top1 - bottom2, top2 - bottom1)

    # The shortest distance between the boxes (Euclidean distance if diagonally separated)
    shortest_distance = math.sqrt(dx**2 + dy**2)

    return shortest_distance <= max_distance

def merge_bboxes(bbox1, bbox2):
    """Merges two bounding boxes into a single encompassing bounding box."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    x_min = min(x1, x2)
    y_min = min(y1, y2)

    x_max = max(x1 + w1, x2 + w2)
    y_max = max(y1 + h1, y2 + h2)

    w_merged = x_max - x_min
    h_merged = y_max - y_min

    return (x_min, y_min, w_merged, h_merged)


# --- Step 3: Optional Bounding Box Joining ---
if 'output_image' in st.session_state and st.session_state['output_image'] is not None and \
   'num_labels' in st.session_state and st.session_state['num_labels'] is not None and \
   'labels' in st.session_state and st.session_state['labels'] is not None and \
   'stats' in st.session_state and st.session_state['stats'] is not None and \
   'min_fiber_area' in st.session_state and st.session_state['min_fiber_area'] is not None:

    st.subheader("Step 3: (Optional) Bounding Box Joining")
    MAX_DISTANCE_TO_JOIN = st.slider(
        'Max Distance to consider Join (pixels)',
        0, 50, 25, # Min, Max, Default value
        key='max_distance_slider'
    )
    joined_boxes_enabled = st.checkbox('Join Nearby Boxes', key='join_boxes_checkbox')

    if joined_boxes_enabled:

        original_image = st.session_state['original_image']
        stats = st.session_state['stats']
        min_fiber_area = st.session_state['min_fiber_area']
        num_labels = st.session_state['num_labels']
        labels = st.session_state['labels']
        output_image_step2 = st.session_state['output_image']

        # Extract initial bounding boxes
        initial_bboxes = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > min_fiber_area:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                initial_bboxes.append((x, y, w, h))

        # Iterative merging algorithm
        merged_bboxes = list(initial_bboxes)
        something_merged = True

        while something_merged:
            something_merged = False
            next_iteration_bboxes = []
            has_been_processed = [False] * len(merged_bboxes)

            for i, bbox1 in enumerate(merged_bboxes):
                if has_been_processed[i]:
                    continue

                found_merge_for_bbox1 = False

                for j, bbox2 in enumerate(merged_bboxes):
                    if i == j or has_been_processed[j]:
                        continue

                    if are_bboxes_close(bbox1, bbox2, MAX_DISTANCE_TO_JOIN):
                        merged_box = merge_bboxes(bbox1, bbox2)
                        next_iteration_bboxes.append(merged_box)
                        has_been_processed[i] = True
                        has_been_processed[j] = True
                        found_merge_for_bbox1 = True
                        something_merged = True
                        break  # Break inner loop as bbox1 has been merged

                if not found_merge_for_bbox1 and not has_been_processed[i]:
                    next_iteration_bboxes.append(bbox1)

            merged_bboxes = next_iteration_bboxes

        st.write(f"Found {len(initial_bboxes)} initial bounding boxes. After merging, {len(merged_bboxes)} combined regions.")

        # Visualization of merged bounding boxes
        image_with_merged_bboxes_visual = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

        # Draw individual fiber highlights (green) on the new visualization image
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > min_fiber_area:
                image_with_merged_bboxes_visual[labels == i] = (0, 255, 0) # Green color

        # Draw the merged bounding boxes (magenta/red) on top
        for bbox in merged_bboxes:
            x, y, w, h = bbox
            cv2.rectangle(image_with_merged_bboxes_visual, (x, y), (x + w, y + h), (255, 0, 255), 2) # Magenta, thickness 2

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(output_image_step2, cv2.COLOR_BGR2RGB), caption='Individual Fiber Detections (Step 2)', use_container_width=True)
        with col2:
            st.image(cv2.cvtColor(image_with_merged_bboxes_visual, cv2.COLOR_BGR2RGB), caption=f'Merged Bounding Boxes (Count: {len(merged_bboxes)}, Max Distance: {MAX_DISTANCE_TO_JOIN})', use_container_width=True)

        st.session_state['merged_bboxes'] = merged_bboxes
        st.session_state['MAX_DISTANCE_TO_JOIN'] = MAX_DISTANCE_TO_JOIN
    else:
        st.info("Check 'Join Nearby Boxes' to group close fibers together.")
else:
    st.info("Please complete Step 2: Fiber Detection to proceed with bounding box joining.")

print("Added optional bounding box joining logic with slider and visualization to main.py")

def process_background_mask(grayscale_image, threshold_value):
    """Applies binary thresholding to remove the background."""
    _, background_removed_mask = cv2.threshold(grayscale_image, threshold_value, 255, cv2.THRESH_BINARY)
    return background_removed_mask

def detect_and_highlight_fibers(original_image, background_mask, min_area):
    """Applies morphological opening, finds connected components, filters by area, and highlights fibers."""
    kernel = np.ones((3,3), np.uint8)
    cleaned_mask = cv2.morphologyEx(background_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask, 8, cv2.CV_32S)

    output_image_with_fibers = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    detected_count = 0
    initial_bboxes_list = [] # List to store bboxes for detected fibers

    for i in range(1, num_labels): # Skip background component (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_area:
            detected_count += 1

            output_image_with_fibers[labels == i] = (0, 255, 0) # Green color

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            initial_bboxes_list.append((x, y, w, h))

            cv2.rectangle(output_image_with_fibers, (x, y), (x + w, y + h), (0, 0, 255), 1) # Red bounding box
            
    return cleaned_mask, num_labels, labels, stats, centroids, initial_bboxes_list, output_image_with_fibers, detected_count

def iterative_merge_bboxes(bboxes_list, max_distance):
    """Iteratively merges close bounding boxes."""
    merged_bboxes = list(bboxes_list)
    something_merged = True

    while something_merged:
        something_merged = False
        next_iteration_bboxes = []
        has_been_processed = [False] * len(merged_bboxes)

        for i, bbox1 in enumerate(merged_bboxes):
            if has_been_processed[i]:
                continue

            found_merge_for_bbox1 = False

            for j, bbox2 in enumerate(merged_bboxes):
                if i == j or has_been_processed[j]:
                    continue

                if are_bboxes_close(bbox1, bbox2, max_distance):
                    merged_box = merge_bboxes(bbox1, bbox2)
                    next_iteration_bboxes.append(merged_box)
                    has_been_processed[i] = True
                    has_been_processed[j] = True
                    found_merge_for_bbox1 = True
                    something_merged = True
                    break

            if not found_merge_for_bbox1 and not has_been_processed[i]:
                next_iteration_bboxes.append(bbox1)

        merged_bboxes = next_iteration_bboxes
    return merged_bboxes

print("Added image processing utility functions to main.py")
