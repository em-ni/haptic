import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

# ── 0. Parse command line ──────────────────────────────────────────
if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
if not os.path.isfile(image_path):
    print(f"Error: file '{image_path}' not found")
    sys.exit(1)

print(f"Input image: {image_path}")

# ── 1. Load the image ──────────────────────────────────────────────
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Could not read '{image_path}'")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, w = img.shape[:2]
print(f"Image size: {w}x{h}")

# ── 2. Detect colored dots by HSV thresholding ─────────────────────

# --- Blue dots (Actual Trajectory) ---
lower_blue = np.array([100, 120, 100])
upper_blue = np.array([130, 255, 255])
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

# --- Red dots (Target Trajectory) ---
lower_red1 = np.array([0, 120, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 120, 100])
upper_red2 = np.array([180, 255, 255])
mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

# --- Green dot (Start) ---
lower_green = np.array([35, 120, 100])
upper_green = np.array([85, 255, 255])
mask_green = cv2.inRange(hsv, lower_green, upper_green)

# --- Black dot (End) ---
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 80, 60])
mask_black = cv2.inRange(hsv, lower_black, upper_black)

def get_dot_centers(mask, min_area=15, max_area=800):
    """Find centroids of detected blobs. If dots are merged, use erosion to separate."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    has_large_blob = False
    for c in contours:
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            M = cv2.moments(c)
            if M['m00'] > 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
                centers.append((cx, cy))
        if area >= max_area:
            has_large_blob = True

    # If too few dots found or large merged blobs exist, try erosion to separate
    if len(centers) < 5 or has_large_blob:
        kernel = np.ones((3, 3), np.uint8)
        best_centers = list(centers)
        for iters in range(1, 6):
            eroded = cv2.erode(mask, kernel, iterations=iters)
            contours_e, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidate_centers = []
            for c in contours_e:
                area = cv2.contourArea(c)
                if area < max_area:
                    M = cv2.moments(c)
                    if M['m00'] > 0:
                        cx = M['m10'] / M['m00']
                        cy = M['m01'] / M['m00']
                        candidate_centers.append((cx, cy))
                    elif len(c) == 1:
                        # Single pixel contour
                        candidate_centers.append((float(c[0][0][0]), float(c[0][0][1])))
            if len(candidate_centers) > len(best_centers):
                best_centers = candidate_centers
            # If we got a good number and more erosion starts losing them, stop
            if len(candidate_centers) > 10 and iters >= 2:
                break
            if len(candidate_centers) == 0 and iters >= 3:
                break
        centers = best_centers

    return np.array(centers) if centers else np.empty((0, 2))

blue_px = get_dot_centers(mask_blue)
red_px = get_dot_centers(mask_red)
green_px = get_dot_centers(mask_green, min_area=30, max_area=1500)
black_px = get_dot_centers(mask_black, min_area=30, max_area=1500)

print(f"Detected blue (actual) dots: {len(blue_px)}")
print(f"Detected red  (target) dots: {len(red_px)}")
print(f"Detected green (start) dots: {len(green_px)}")
print(f"Detected black (end)   dots: {len(black_px)}")

# ── 2b. Filter out legend dots ──────────────────────────────────────
# Legend dots are spatially isolated from the main trajectory data.
# Strategy: remove dots that have no neighbor within a reasonable distance
# (i.e., they sit alone, likely in the legend area).
def filter_legend_dots(centers, min_neighbor_dist=80):
    """Remove dots that are spatially isolated (no nearby neighbors)."""
    if len(centers) < 3:
        return centers
    from scipy.spatial.distance import cdist
    D = cdist(centers, centers)
    np.fill_diagonal(D, np.inf)
    nearest = D.min(axis=1)
    # A dot is considered "isolated" if its nearest neighbor is much farther
    # than typical. Use median nearest-neighbor distance as reference.
    med_nn = np.median(nearest)
    # Keep dots whose nearest neighbor is within a generous multiple of the
    # median, OR within min_neighbor_dist pixels (whichever is larger)
    threshold = max(med_nn * 5.0, min_neighbor_dist)
    keep = nearest < threshold
    removed = np.sum(~keep)
    if removed > 0:
        for i in np.where(~keep)[0]:
            print(f"  Filtered legend dot at ({centers[i,0]:.0f},{centers[i,1]:.0f}), nearest neighbor={nearest[i]:.0f}px")
    return centers[keep]

if len(blue_px) > 3:
    blue_px = filter_legend_dots(blue_px)
if len(red_px) > 3:
    red_px = filter_legend_dots(red_px)

print(f"After filtering — blue: {len(blue_px)}, red: {len(red_px)}")

# ── 3. Auto-detect trajectory shape: CIRCLE vs LINE ───────────────
# Use the raw red mask to detect shape. If the red pixels form a blob
# whose bounding-box aspect and PCA eigenratio are high → circle,
# otherwise → line.
red_pixels = np.column_stack(np.where(mask_red > 0))  # (row, col)
if len(red_pixels) < 10:
    raise RuntimeError("Not enough red pixels detected")
red_px_for_shape = red_pixels[:, ::-1].astype(float)  # (x, y)
red_xmin_px, red_xmax_px = red_px_for_shape[:, 0].min(), red_px_for_shape[:, 0].max()
red_ymin_px, red_ymax_px = red_px_for_shape[:, 1].min(), red_px_for_shape[:, 1].max()
span_x = red_xmax_px - red_xmin_px
span_y = red_ymax_px - red_ymin_px
aspect = min(span_x, span_y) / max(span_x, span_y) if max(span_x, span_y) > 0 else 1.0

red_centered = red_px_for_shape - red_px_for_shape.mean(axis=0)
cov = np.cov(red_centered.T)
eigvals = np.linalg.eigvalsh(cov)
eigenratio = eigvals.min() / eigvals.max() if eigvals.max() > 0 else 1.0

print(f"\nShape detection — aspect ratio: {aspect:.3f}, PCA eigenratio: {eigenratio:.4f}")

is_circle = eigenratio > 0.15 and aspect > 0.4
shape_name = "CIRCLE" if is_circle else "LINE"
print(f"Detected shape: {shape_name}")

# For LINE: get the two extreme red endpoints directly from the red mask
# instead of separating merged dots. The target is defined by a straight line
# between these two endpoints.
if not is_circle:
    # The two endpoints are the red pixels farthest apart
    from scipy.spatial.distance import cdist
    # Subsample if too many pixels for speed
    pts = red_px_for_shape
    if len(pts) > 2000:
        idx_sub = np.random.choice(len(pts), 2000, replace=False)
        pts_sub = pts[idx_sub]
    else:
        pts_sub = pts
    dists_mat = cdist(pts_sub, pts_sub)
    i_max, j_max = np.unravel_index(np.argmax(dists_mat), dists_mat.shape)
    red_endpoint1_px = pts_sub[i_max]  # (x, y) in pixel
    red_endpoint2_px = pts_sub[j_max]
    print(f"Red line endpoints (px): ({red_endpoint1_px[0]:.0f},{red_endpoint1_px[1]:.0f}) — ({red_endpoint2_px[0]:.0f},{red_endpoint2_px[1]:.0f})")
    # Override red_px with just the two endpoints (used later only for range)
    red_xmin_px = min(red_endpoint1_px[0], red_endpoint2_px[0])
    red_xmax_px = max(red_endpoint1_px[0], red_endpoint2_px[0])
    red_ymin_px = min(red_endpoint1_px[1], red_endpoint2_px[1])
    red_ymax_px = max(red_endpoint1_px[1], red_endpoint2_px[1])
    span_x = red_xmax_px - red_xmin_px
    span_y = red_ymax_px - red_ymin_px

# ── 4. Calibrate pixel → mm ────────────────────────────────────────
# Strategy: detect the plot axes frame, detect tick marks, estimate mm/px scale.
print(f"\nRed dots pixel range: x=[{red_xmin_px:.0f}, {red_xmax_px:.0f}], y=[{red_ymin_px:.0f}, {red_ymax_px:.0f}]")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
lines_hough = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=5)
h_lines = []
v_lines = []
if lines_hough is not None:
    for line in lines_hough:
        x1, y1, x2, y2 = line[0]
        if abs(y2 - y1) < 5:
            h_lines.append((min(x1,x2), max(x1,x2), (y1+y2)//2))
        elif abs(x2 - x1) < 5:
            v_lines.append((min(y1,y2), max(y1,y2), (x1+x2)//2))

h_lines.sort(key=lambda l: l[1]-l[0], reverse=True)
v_lines.sort(key=lambda l: l[1]-l[0], reverse=True)
h_ys = sorted(set(l[2] for l in h_lines[:10]))
v_xs = sorted(set(l[2] for l in v_lines[:10]))

plot_top_px = h_ys[0]
plot_bot_px = h_ys[-1]
plot_left_px = v_xs[0]
plot_right_px = v_xs[-1]
plot_w_px = plot_right_px - plot_left_px
plot_h_px = plot_bot_px - plot_top_px
print(f"Plot frame (px): left={plot_left_px}, right={plot_right_px}, top={plot_top_px}, bottom={plot_bot_px}")

# Detect X-axis tick marks: short vertical lines near the bottom of the plot
tick_candidates_x = []
if lines_hough is not None:
    for line in lines_hough:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) < 3:  # vertical
            length = abs(y2 - y1)
            mid_y = (y1 + y2) / 2
            # Ticks are near the bottom axis line
            if 5 < length < 30 and abs(mid_y - plot_bot_px) < 20:
                tick_candidates_x.append((x1 + x2) / 2)

# Detect Y-axis tick marks: short horizontal lines near the left of the plot
tick_candidates_y = []
if lines_hough is not None:
    for line in lines_hough:
        x1, y1, x2, y2 = line[0]
        if abs(y2 - y1) < 3:  # horizontal
            length = abs(x2 - x1)
            mid_x = (x1 + x2) / 2
            if 5 < length < 30 and abs(mid_x - plot_left_px) < 20:
                tick_candidates_y.append((y1 + y2) / 2)

# Cluster tick positions (they may have duplicates within a few pixels)
def cluster_ticks(positions, min_gap=15):
    if not positions:
        return []
    positions = sorted(positions)
    clusters = [[positions[0]]]
    for p in positions[1:]:
        if p - clusters[-1][-1] < min_gap:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [np.mean(c) for c in clusters]

x_ticks_px = cluster_ticks(tick_candidates_x)
y_ticks_px = cluster_ticks(tick_candidates_y)
print(f"Detected {len(x_ticks_px)} X-ticks, {len(y_ticks_px)} Y-ticks")

# Estimate mm/px from tick spacing.
# Matplotlib ticks are at regular mm intervals (0.5 or 1.0 typically).
# The spacing in pixels between consecutive ticks should be constant.
if len(x_ticks_px) >= 2:
    x_tick_spacing_px = np.mean(np.diff(sorted(x_ticks_px)))
    # Try tick spacings of 0.5 and 1.0 mm — pick the one giving rounder axis limits
    best_tick_mm = 1.0
    best_score = 999
    for tick_mm in [0.5, 1.0, 1.5, 2.0]:
        sx = tick_mm / x_tick_spacing_px
        # What would the axis limits be?
        xmin_mm = -(sorted(x_ticks_px)[0] - plot_left_px) * sx
        xmax_mm = (plot_right_px - sorted(x_ticks_px)[0]) * sx
        # Check roundness
        def roundness(v, step=0.5):
            return abs(v - round(v / step) * step)
        score = roundness(xmin_mm) + roundness(xmax_mm)
        if score < best_score:
            best_score = score
            best_tick_mm = tick_mm
    scale_x = best_tick_mm / x_tick_spacing_px
    print(f"X tick spacing: {x_tick_spacing_px:.1f} px = {best_tick_mm} mm → scale_x = {scale_x:.6f} mm/px")
else:
    # Fallback for circle: use bounding box of red dots = diameter
    scale_x = 4.0 / span_x

if len(y_ticks_px) >= 2:
    y_tick_spacing_px = np.mean(np.diff(sorted(y_ticks_px)))
    best_tick_mm_y = 0.5
    best_score = 999
    for tick_mm in [0.5, 1.0, 1.5, 2.0]:
        sy = tick_mm / y_tick_spacing_px
        score_val = 0
        for ytp in sorted(y_ticks_px):
            val = -(ytp - (plot_top_px + plot_bot_px)/2) * sy
            score_val += roundness(val, 0.5)
        score_val /= len(y_ticks_px)
        if score_val < best_score:
            best_score = score_val
            best_tick_mm_y = tick_mm
    scale_y = best_tick_mm_y / y_tick_spacing_px
    print(f"Y tick spacing: {y_tick_spacing_px:.1f} px = {best_tick_mm_y} mm → scale_y = {scale_y:.6f} mm/px")
else:
    scale_y = scale_x

# Find the origin (0,0) in pixel space.
# For ticks at regular intervals, one tick should be at x=0 and y=0.
# The origin is the tick closest to the center of the data.
if is_circle:
    # Circle center = origin
    center_px_x = (red_xmin_px + red_xmax_px) / 2
    center_px_y = (red_ymin_px + red_ymax_px) / 2
    # Snap to nearest tick
    if x_ticks_px:
        center_px_x = min(x_ticks_px, key=lambda t: abs(t - center_px_x))
    if y_ticks_px:
        center_px_y = min(y_ticks_px, key=lambda t: abs(t - center_px_y))
else:
    # Line: origin is somewhere on the axes. Find it from ticks.
    data_cx = (red_xmin_px + red_xmax_px) / 2
    data_cy = (red_ymin_px + red_ymax_px) / 2
    if x_ticks_px:
        center_px_x = min(x_ticks_px, key=lambda t: abs(t - data_cx))
    else:
        center_px_x = data_cx
    if y_ticks_px:
        center_px_y = min(y_ticks_px, key=lambda t: abs(t - data_cy))
    else:
        center_px_y = data_cy

print(f"Origin (0,0) at pixel: ({center_px_x:.0f}, {center_px_y:.0f})")
print(f"Scale: x={scale_x:.6f} mm/px, y={scale_y:.6f} mm/px")

# ── 5. Convert pixel → mm ─────────────────────────────────────────
def px_to_mm(points_px):
    """Convert pixel coordinates to mm coordinates."""
    mm = np.zeros_like(points_px)
    mm[:, 0] = (points_px[:, 0] - center_px_x) * scale_x
    mm[:, 1] = -(points_px[:, 1] - center_px_y) * scale_y  # flip y
    return mm

blue_mm = px_to_mm(blue_px)
red_mm = px_to_mm(red_px)

print(f"\nRed (target) mm range:  x=[{red_mm[:,0].min():.2f}, {red_mm[:,0].max():.2f}], y=[{red_mm[:,1].min():.2f}, {red_mm[:,1].max():.2f}]")
print(f"Blue (actual) mm range: x=[{blue_mm[:,0].min():.2f}, {blue_mm[:,0].max():.2f}], y=[{blue_mm[:,1].min():.2f}, {blue_mm[:,1].max():.2f}]")

green_mm = None
black_mm = None

# Filter green/black dots: must be inside the plot frame (pixel space)
# This excludes title text, axis labels, etc. that are detected as black blobs.
def filter_inside_plot_frame(px_pts, left, right, top, bottom, pad=5):
    """Keep only points inside the plot frame with a small padding."""
    if len(px_pts) == 0:
        return px_pts
    inside = (
        (px_pts[:, 0] > left - pad) & (px_pts[:, 0] < right + pad) &
        (px_pts[:, 1] > top - pad) & (px_pts[:, 1] < bottom + pad)
    )
    return px_pts[inside]

green_px_valid = filter_inside_plot_frame(green_px, plot_left_px, plot_right_px, plot_top_px, plot_bot_px)
black_px_valid = filter_inside_plot_frame(black_px, plot_left_px, plot_right_px, plot_top_px, plot_bot_px)

if len(green_px_valid) > 0:
    green_mm_all = px_to_mm(green_px_valid)
    data_center = np.mean(np.vstack([blue_mm, red_mm]), axis=0)
    dists = np.linalg.norm(green_mm_all - data_center, axis=1)
    best_idx = np.argmin(dists)
    green_mm = green_mm_all[best_idx:best_idx+1]
    print(f"Green (start) mm: ({green_mm[0,0]:.2f}, {green_mm[0,1]:.2f})")
else:
    print("Green (start): no valid dot found within plot frame")

if len(black_px_valid) > 0:
    black_mm_all = px_to_mm(black_px_valid)
    data_center = np.mean(np.vstack([blue_mm, red_mm]), axis=0)
    dists = np.linalg.norm(black_mm_all - data_center, axis=1)
    best_idx = np.argmin(dists)
    black_mm = black_mm_all[best_idx:best_idx+1]
    print(f"Black (end)   mm: ({black_mm[0,0]:.2f}, {black_mm[0,1]:.2f})")
else:
    print("Black (end):   no valid dot found within plot frame")

# ── 6. Order points along the trajectory ───────────────────────────

def order_by_angle_ccw(points, start_angle=0.0):
    """Order points counterclockwise by angle from center (0,0)."""
    angles = np.arctan2(points[:, 1], points[:, 0])
    angles = (angles - start_angle) % (2 * np.pi)
    order = np.argsort(angles)
    return points[order]

def order_along_line(points, start_point):
    """Order points by greedy nearest-neighbour walk starting from start_point."""
    remaining = list(range(len(points)))
    # Find closest point to start_point
    dists = np.linalg.norm(points[remaining] - start_point, axis=1)
    first = remaining[np.argmin(dists)]
    ordered_idx = [first]
    remaining.remove(first)
    while remaining:
        cur = points[ordered_idx[-1]]
        dists = np.linalg.norm(points[remaining] - cur, axis=1)
        nxt = remaining[np.argmin(dists)]
        ordered_idx.append(nxt)
        remaining.remove(nxt)
    return points[ordered_idx]

if is_circle:
    # Target: starts at rightmost point (angle=0), counterclockwise
    red_ordered = order_by_angle_ccw(red_mm, start_angle=0.0)

    # Actual: starts at green dot
    if green_mm is not None:
        start_angle_actual = np.arctan2(green_mm[0, 1], green_mm[0, 0])
    else:
        start_angle_actual = 0.0
    blue_ordered = order_by_angle_ccw(blue_mm, start_angle=start_angle_actual)
    print(f"\nActual start angle: {np.degrees(start_angle_actual):.1f}°")
else:
    # LINE: target is defined by two endpoints → straight line equation.
    # Convert the two red endpoints to mm
    ep1_mm = px_to_mm(np.array([red_endpoint1_px]))[0]
    ep2_mm = px_to_mm(np.array([red_endpoint2_px]))[0]
    # Target start = bottom-right endpoint (largest x - y)
    if (ep1_mm[0] - ep1_mm[1]) > (ep2_mm[0] - ep2_mm[1]):
        line_start, line_end = ep1_mm, ep2_mm
    else:
        line_start, line_end = ep2_mm, ep1_mm
    print(f"\nTarget line: ({line_start[0]:.2f}, {line_start[1]:.2f}) → ({line_end[0]:.2f}, {line_end[1]:.2f})")

    # Actual: starts at green dot
    if green_mm is not None:
        blue_ordered = order_along_line(blue_mm, green_mm[0])
    else:
        blue_ordered = order_along_line(blue_mm, line_start)

    # Generate dense red_ordered along the line for plotting
    n_target_pts = max(len(blue_ordered), 100)
    t_vals = np.linspace(0, 1, n_target_pts)
    red_ordered = np.column_stack([
        line_start[0] + t_vals * (line_end[0] - line_start[0]),
        line_start[1] + t_vals * (line_end[1] - line_start[1]),
    ])

print(f"Ordered red  points: {len(red_ordered)}")
print(f"Ordered blue points: {len(blue_ordered)}")

# ── 7. Match actual points to nearest target point ─────────────────

if is_circle:
    # Match by angle
    red_abs_angles = np.arctan2(red_ordered[:, 1], red_ordered[:, 0])
    blue_abs_angles = np.arctan2(blue_ordered[:, 1], blue_ordered[:, 0])

    errors = []
    matched_pairs = []
    for bpt, bang in zip(blue_ordered, blue_abs_angles):
        angle_diffs = np.abs(np.arctan2(
            np.sin(red_abs_angles - bang), np.cos(red_abs_angles - bang)))
        closest_idx = np.argmin(angle_diffs)
        rpt = red_ordered[closest_idx]
        err = np.linalg.norm(bpt - rpt)
        errors.append(err)
        matched_pairs.append((bpt, rpt))
else:
    # LINE: for each blue point, find the closest point on the line segment
    A = line_start
    B = line_end
    AB = B - A
    AB_len2 = np.dot(AB, AB)

    errors = []
    matched_pairs = []
    for bpt in blue_ordered:
        # Project bpt onto line AB, clamped to [0,1]
        t = np.clip(np.dot(bpt - A, AB) / AB_len2, 0.0, 1.0)
        closest_on_line = A + t * AB
        err = np.linalg.norm(bpt - closest_on_line)
        errors.append(err)
        matched_pairs.append((bpt, closest_on_line))

errors = np.array(errors)

# ── 8. Calculate RMSE ──────────────────────────────────────────────
rmse = np.sqrt(np.mean(errors**2))
mae = np.mean(errors)
max_err = np.max(errors)
std_err = np.std(errors)

print("\n" + "="*60)
print(f"  RESULTS — {shape_name} (extracted from image)")
print("="*60)
print(f"  Number of matched point pairs: {len(errors)}")
print(f"  RMSE:          {rmse:.4f} mm")
print(f"  MAE:           {mae:.4f} mm")
print(f"  Max error:     {max_err:.4f} mm")
print(f"  Std error:     {std_err:.4f} mm")
print("="*60)

# ── 9. Verification plot ──────────────────────────────────────────
fig = plt.figure(figsize=(22, 6.5))

# ── Panel 1: Original image ──
ax1 = fig.add_subplot(1, 3, 1)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
ax1.imshow(img_rgb)
ax1.set_title('Original Image', fontsize=13, fontweight='bold')
ax1.axis('off')

# ── Panel 2: Extracted trajectories with rainbow-colored error lines ──
ax2 = fig.add_subplot(1, 3, 2)

cmap = plt.cm.rainbow
norm = plt.Normalize(vmin=0, vmax=len(matched_pairs) - 1)

for idx, (bpt, rpt) in enumerate(matched_pairs):
    color = cmap(norm(idx))
    ax2.plot([bpt[0], rpt[0]], [bpt[1], rpt[1]], '-', color=color, alpha=0.5, linewidth=1.2)

for idx, (bpt, rpt) in enumerate(matched_pairs):
    color = cmap(norm(idx))
    ax2.plot(bpt[0], bpt[1], 'o', color=color, markersize=5, markeredgecolor='blue',
             markeredgewidth=0.4, zorder=3)
    ax2.plot(rpt[0], rpt[1], 'o', color=color, markersize=5, markeredgecolor='red',
             markeredgewidth=0.4, zorder=3)

if green_mm is not None:
    ax2.plot(green_mm[0, 0], green_mm[0, 1], 'go', markersize=12, zorder=5, label='Start')
if black_mm is not None:
    ax2.plot(black_mm[0, 0], black_mm[0, 1], 'ko', markersize=12, zorder=5, label='End')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label('Point index along trajectory', fontsize=9)

# Set axis limits based on data extent with padding
all_pts = np.vstack([red_ordered, blue_ordered])
pad = 0.5
ax2.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
ax2.set_ylim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)

ax2.set_xlabel('X Position (mm)')
ax2.set_ylabel('Y Position (mm)')
ax2.set_title(f'Extracted Trajectories — RMSE: {rmse:.4f} mm', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9, loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

# ── Panel 3: Error along the trajectory with matching rainbow colors ──
ax3 = fig.add_subplot(1, 3, 3)

for idx, err_val in enumerate(errors):
    color = cmap(norm(idx))
    ax3.bar(idx, err_val, color=color, width=1.0, edgecolor='none')

ax3.plot(range(len(errors)), errors, 'k-', linewidth=0.8, alpha=0.6)
ax3.axhline(mae, color='grey', linestyle='--', linewidth=1.2, label=f'MAE = {mae:.4f} mm')
ax3.axhline(rmse, color='black', linestyle='-', linewidth=1.2, label=f'RMSE = {rmse:.4f} mm')
ax3.set_xlabel('Point index (along trajectory)')
ax3.set_ylabel('Error (mm)')
ax3.set_title('Point-wise Tracking Error', fontsize=13, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
out_name = os.path.splitext(os.path.basename(image_path))[0] + '_rmse.png'
out_path = os.path.join('results', out_name)
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved to {out_path}")
