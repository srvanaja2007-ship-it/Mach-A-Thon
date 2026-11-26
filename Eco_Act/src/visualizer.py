import os
import cv2
import json
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _iter_landmark_pixels(landmark_info, w, h):
    if not landmark_info:
        return []
    landmarks = landmark_info.get('landmarks', []) if isinstance(landmark_info, dict) else landmark_info
    coord_fn = landmark_info.get('coord_to_pixel') if isinstance(landmark_info, dict) else None
    pts = []
    for lm in landmarks:
        if 'pixel_x' in lm and 'pixel_y' in lm:
            px, py = int(lm['pixel_x']), int(lm['pixel_y'])
        elif coord_fn is not None and 'lat' in lm and 'lon' in lm:
            px, py = coord_fn(lm['lat'], lm['lon'], w=w, h=h)
        elif 'x_norm' in lm and 'y_norm' in lm:
            px, py = int(lm['x_norm'] * w), int(lm['y_norm'] * h)
        else:
            continue
        pts.append((lm['name'], px, py))
    return pts


def render_optical_vs_nightlights_comparison(optical_path, ntl_img, output_path, region_name="Tumkur", landmark_info=None):
    """
    Renders side-by-side comparison between daytime optical satellite imagery (with landmarks)
    and nighttime satellite radiance (VIIRS). Skips cleanly if optical_path is missing.
    """
    if not optical_path or not os.path.exists(optical_path):
        print(f"[Visualizer] Skipping optical comparison (no optical image at '{optical_path}').")
        return
    opt_bgr = cv2.imread(optical_path)
    if opt_bgr is None:
        return
    opt_rgb = cv2.cvtColor(opt_bgr, cv2.COLOR_BGR2RGB)

    ntl_colored = cv2.applyColorMap(np.clip(ntl_img, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    ntl_rgb = cv2.cvtColor(ntl_colored, cv2.COLOR_BGR2RGB)

    for name, px, py in _iter_landmark_pixels(landmark_info, ntl_img.shape[1], ntl_img.shape[0]):
        cv2.circle(ntl_rgb, (px, py), 5, (0, 255, 255), -1)
        cv2.putText(ntl_rgb, name[:14], (px + 6, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='#0d1117')
    ax1.imshow(opt_rgb)
    ax1.set_title("Daytime Optical Satellite (High-Res + Landmarks)", color='#58a6ff', fontsize=14, fontweight='bold', pad=10)
    ax1.axis('off')

    ax2.imshow(ntl_rgb)
    ax2.set_title("Nighttime Satellite Radiance (VIIRS - Economic Activity)", color='#f0883e', fontsize=14, fontweight='bold', pad=10)
    ax2.axis('off')

    plt.suptitle(f"{region_name}: Optical Ground Truth vs Satellite Night Lights",
                 color='#f0f6fc', fontsize=16, fontweight='bold', y=0.96)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[Visualizer] Saved Optical vs Nightlights comparison to: {output_path}")


def render_plot_timeline(images_hist, years_hist, images_future, years_future, output_path, region_name="Tumkur", landmark_info=None):
    """
    Renders multi-panel timeline comparing representative historical years with future AI predictions.
    """
    unique_years = sorted(set(years_hist))
    if len(unique_years) <= 6:
        key_years = unique_years
    else:
        sel = np.linspace(0, len(unique_years) - 1, 6, dtype=int)
        key_years = [unique_years[i] for i in sel]

    hist_sample_imgs = []
    hist_sample_labels = []

    for ky in key_years:
        indices = [idx for idx, y in enumerate(years_hist) if y == ky]
        if indices:
            mid_idx = indices[len(indices) // 2]
            hist_sample_imgs.append(images_hist[mid_idx])
            hist_sample_labels.append(f"{ky} [VIIRS]")

    future_imgs = [item[1] for item in images_future]
    future_labels = [f"{item[0]} [AI PREDICTED]" for item in images_future]

    all_imgs = hist_sample_imgs + future_imgs
    all_labels = hist_sample_labels + future_labels
    total = len(all_imgs)

    cols = min(3, total)
    rows = (total + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5.5 * rows), facecolor='#0d1117')
    axes = np.array(axes).reshape(rows, cols)

    for idx in range(rows * cols):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]
        if idx < total:
            img = all_imgs[idx]
            lbl = all_labels[idx]
            is_pred = "PREDICTED" in lbl
            color = '#ff7b72' if is_pred else '#58a6ff'

            ax.imshow(img, cmap='inferno', vmin=0, vmax=255)
            ax.set_title(lbl, color=color, fontsize=12, fontweight='bold', pad=8)
            ax.axis('off')

            for name, px, py in _iter_landmark_pixels(landmark_info, img.shape[1], img.shape[0])[:5]:
                ax.plot(px, py, marker='o', markersize=4, color='#00ffcc')
                if idx == total - 1 or idx == 0:
                    ax.text(px + 6, py - 4, name[:12], color='#00ffcc', fontsize=7, fontweight='bold')
        else:
            ax.axis('off')

    plt.suptitle(f"{region_name} Spatio-Temporal Development: Satellite History & AI Forecast",
                 color='#f0f6fc', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[Visualizer] Saved timeline progression to: {output_path}")


def render_plot_metrics(hist_df, pred_df, zones, output_path, region_name="Tumkur", base_gddp=32000.0):
    """
    Renders 4 analytical plots showing SOL, GDDP forecast, zone growth, and economic velocity.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 11), facecolor='#0d1117')
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
        ax.grid(True, linestyle='--', alpha=0.3, color='#8b949e')

    t_hist = hist_df['timestamp'].values
    ax1.plot(t_hist, hist_df['sum_of_lights'], color='#30363d', alpha=0.6, label='Raw Radiant Flux')
    ax1.plot(t_hist, hist_df['sol_trend_smooth'], color='#58a6ff', linewidth=2.5, label='Smoothed Trend')
    last_t = t_hist[-1]
    last_sol = hist_df['sol_trend_smooth'].iloc[-1]
    pred_t = [last_t] + list(pred_df['year'].values)
    pred_sol = [last_sol] + list(pred_df['predicted_sum_of_lights'].values)
    ax1.plot(pred_t, pred_sol, color='#f0883e', linestyle='--', linewidth=2.5, label='AI Forecast Trend')
    ax1.set_title("Radiant Flux (Historical VIIRS + AI Forecast)", color='#f0f6fc', fontsize=12, fontweight='bold')
    ax1.set_ylabel("Sum of Lights Flux", color='#8b949e')
    ax1.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#f0f6fc')

    base_sol = max(1.0, hist_df['sol_trend_smooth'].iloc[0])
    hist_gddp = base_gddp * ((hist_df['sol_trend_smooth'] / base_sol) ** 0.75)
    ax2.plot(t_hist, hist_gddp, color='#3fb950', linewidth=2.5, label='Nowcasted GDDP (₹ Cr)')
    pred_gddp = [hist_gddp.iloc[-1]] + list(pred_df['predicted_gddp_crores'].values)
    ax2.plot(pred_t, pred_gddp, color='#d29922', linestyle='--', linewidth=2.5, label='AI Projected GDDP')

    ci_low = [hist_gddp.iloc[-1]] + list(pred_df['confidence_interval_low_gddp'].values)
    ci_high = [hist_gddp.iloc[-1]] + list(pred_df['confidence_interval_high_gddp'].values)
    ax2.fill_between(pred_t, ci_low, ci_high, color='#d29922', alpha=0.2, label='95% Confidence Band')
    ax2.set_title(f"Estimated {region_name} Output / GDDP (₹ Crores)", color='#f0f6fc', fontsize=12, fontweight='bold')
    ax2.set_ylabel("₹ in Crores", color='#8b949e')
    ax2.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#f0f6fc')

    top_zones = zones[:6]
    names = [z['name'][:22] for z in top_zones][::-1]
    growths = [z['growth_pct'] for z in top_zones][::-1]
    colors = ['#3fb950' if g > 20 else '#58a6ff' for g in growths]
    ax3.barh(names, growths, color=colors, height=0.6)
    ax3.set_title("Projected Growth Velocity by Zone (%)", color='#f0f6fc', fontsize=12, fontweight='bold')
    ax3.set_xlabel("% Projected Radiance Expansion", color='#8b949e')

    hist_idx = (hist_gddp / base_gddp) * 35.0
    pred_idx = [hist_idx.iloc[-1]] + list(pred_df['economic_activity_index'].values)
    ax4.plot(t_hist, hist_idx, color='#bc8cff', linewidth=2.5, label='Historical Econ Index')
    ax4.plot(pred_t, pred_idx, color='#f778ba', linestyle='--', linewidth=2.5, label='Projected Index')
    ax4.set_title("Economic Activity Index (0 - 100)", color='#f0f6fc', fontsize=12, fontweight='bold')
    ax4.set_ylabel("Index Score", color='#8b949e')
    ax4.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#f0f6fc')

    plt.suptitle(f"{region_name} Economic Nowcasting & Growth Projections (Satellite AI Model)", color='#f0f6fc', fontsize=16, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[Visualizer] Saved metrics forecast plot to: {output_path}")

def generate_interactive_html_dashboard(hist_df, pred_df, images_hist, timestamps_hist, images_future, years_future, zones, optical_path, output_path, region_name="Tumkur", base_gddp=32000.0, table_filename=None):
    """
    Creates standalone interactive HTML dashboard featuring:
    - Side-by-side Optical Satellite View (or baseline frame) vs Night Lights
    - Scrubbable Time Slider across historical + predicted frames
    - Live GDDP Nowcasting & Opportunity Zones table
    """
    n_samples = min(24, len(images_hist))
    sample_indices = np.linspace(0, len(images_hist) - 1, n_samples, dtype=int)
    sampled_imgs = [images_hist[i] for i in sample_indices]
    sampled_labels = [f"{int(timestamps_hist[i])}-{int(round((timestamps_hist[i] % 1) * 12)) + 1:02d}" for i in sample_indices]

    for yr, fut_img in images_future:
        sampled_imgs.append(fut_img)
        sampled_labels.append(f"{yr} (AI PREDICTED)")

    encoded_frames = []
    for img in sampled_imgs:
        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(img_u8, cv2.COLORMAP_INFERNO)
        _, buf = cv2.imencode('.jpg', colored, [cv2.IMWRITE_JPEG_QUALITY, 80])
        encoded_frames.append(base64.b64encode(buf).decode('utf-8'))

    has_optical = bool(optical_path and os.path.exists(optical_path))
    opt_bgr = cv2.imread(optical_path) if has_optical else None
    if opt_bgr is not None:
        _, opt_buf = cv2.imencode('.jpg', opt_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        opt_b64 = base64.b64encode(opt_buf).decode('utf-8')
        left_title = "Optical Daytime Satellite (Ground Truth)"
        left_badge = "OPTICAL REFERENCE"
        left_desc = f"Ground truth optical reference for {region_name}."
    else:
        opt_b64 = encoded_frames[0]
        left_title = f"Baseline Night Lights ({sampled_labels[0]})"
        left_badge = "BASELINE FRAME"
        left_desc = f"Earliest baseline satellite frame for {region_name} to compare against forecast."

    chart_years_clean = [f"{int(t)}-{int(round((t % 1) * 12)) + 1:02d}" for t in hist_df['timestamp']]
    base_sol = max(1.0, hist_df['sol_trend_smooth'].iloc[0])
    hist_gddp = list(np.round(base_gddp * ((hist_df['sol_trend_smooth'] / base_sol) ** 0.75), 1))

    future_labels_chart = [str(yr) for yr in years_future]
    future_gddp_chart = list(np.round(pred_df['predicted_gddp_crores'], 1))

    total_labels = chart_years_clean + future_labels_chart
    total_gddp = hist_gddp + future_gddp_chart

    latest_gddp = total_gddp[-1]
    baseline_gddp = total_gddp[0]
    gddp_growth = round(((latest_gddp - baseline_gddp) / max(1.0, baseline_gddp)) * 100.0, 1)
    start_yr = int(hist_df['year'].iloc[0])
    end_yr = int(hist_df['year'].iloc[-1])
    fut_end_yr = int(years_future[-1])
    res_h, res_w = images_hist.shape[1], images_hist.shape[2]

    zones_html = ""
    for z in zones[:8]:
        color_class = "badge-success" if z['growth_pct'] > 20 else "badge-warning"
        zones_html += f"""
        <tr>
            <td><strong>{z['name']}</strong></td>
            <td><span class="badge badge-info">{z['type'].upper()}</span></td>
            <td>+{z['growth_pct']:.1f}%</td>
            <td><span class="badge {color_class}">{z['status']}</span></td>
            <td><small>{z['recommended_action']}</small></td>
        </tr>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tumkur & Outskirts: 10-Year Satellite Economic Intelligence</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-dark: #090d16;
            --card-bg: rgba(22, 27, 34, 0.88);
            --border-color: #30363d;
            --accent-blue: #58a6ff;
            --accent-orange: #f0883e;
            --accent-green: #3fb950;
            --text-main: #f0f6fc;
            --text-muted: #8b949e;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-main);
            margin: 0;
            padding: 24px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 16px;
            margin-bottom: 24px;
        }}
        .header h1 {{ margin: 0; font-size: 25px; color: var(--accent-blue); }}
        .header p {{ margin: 4px 0 0 0; color: var(--text-muted); font-size: 14px; }}
        .badge-track {{
            background: rgba(88, 166, 255, 0.15);
            border: 1px solid var(--accent-blue);
            color: var(--accent-blue);
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 18px;
            backdrop-filter: blur(8px);
        }}
        .card-label {{ font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }}
        .card-value {{ font-size: 26px; font-weight: bold; margin-top: 6px; color: var(--text-main); }}
        .card-sub {{ font-size: 12px; margin-top: 4px; color: var(--accent-green); }}

        .comparison-layout {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 24px;
        }}
        .viewer-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 18px;
            text-align: center;
        }}
        .image-frame {{
            width: 100%;
            height: auto;
            max-height: 440px;
            object-fit: contain;
            border-radius: 8px;
            border: 1px solid #484f58;
            box-shadow: 0 8px 24px rgba(0,0,0,0.6);
            margin: 12px auto;
            display: block;
        }}
        .slider-control {{
            margin-top: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 16px;
        }}
        input[type=range] {{
            width: 60%;
            cursor: pointer;
            accent-color: var(--accent-blue);
        }}
        .year-display {{
            font-size: 18px;
            font-weight: bold;
            color: var(--accent-orange);
            min-width: 120px;
        }}
        .tag-pill {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: bold;
        }}
        .tag-hist {{ background: rgba(88, 166, 255, 0.2); color: var(--accent-blue); }}
        .tag-pred {{ background: rgba(240, 136, 62, 0.2); color: var(--accent-orange); }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        th, td {{
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        th {{ color: var(--text-muted); font-weight: 600; }}
        .badge {{ padding: 3px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }}
        .badge-success {{ background: rgba(63, 185, 80, 0.15); color: var(--accent-green); border: 1px solid var(--accent-green); }}
        .badge-warning {{ background: rgba(240, 136, 62, 0.15); color: var(--accent-orange); border: 1px solid var(--accent-orange); }}
        .badge-info {{ background: rgba(88, 166, 255, 0.15); color: var(--accent-blue); border: 1px solid var(--accent-blue); }}
        .header-actions {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap; justify-content: flex-end; }}
        .nav-btn {{
            display: inline-flex; align-items: center; gap: 6px;
            background: linear-gradient(135deg, #1f6feb, #388bfd);
            color: #ffffff; text-decoration: none; font-weight: 600; font-size: 13px;
            padding: 8px 16px; border-radius: 8px; border: 1px solid rgba(88, 166, 255, 0.5);
            box-shadow: 0 2px 10px rgba(31, 111, 235, 0.3); transition: transform 0.15s, box-shadow 0.15s;
        }}
        .nav-btn:hover {{ transform: translateY(-1px); box-shadow: 0 4px 14px rgba(56, 139, 253, 0.45); }}
    </style>
</head>
<body>

    <div class="header">
        <div>
            <h1>Orbit to Economy: {region_name} (Spatio-Temporal AI)</h1>
            <p>Satellite Night Lights (VIIRS) Economic Nowcasting &amp; {fut_end_yr} Projections</p>
        </div>
        <div class="header-actions">
            <div class="badge-track">Track A: Read the Planet &bull; Boeing Aeromania '26</div>
            {f'<a href="{table_filename}" class="nav-btn">&#128202; View Predictions &amp; Table &rarr;</a>' if table_filename else ''}
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <div class="card-label">Baseline GDDP ({start_yr})</div>
            <div class="card-value">₹{base_gddp:,.0f} Cr</div>
            <div class="card-sub">Baseline Economic Reference</div>
        </div>
        <div class="card">
            <div class="card-label">Nowcasted Output ({fut_end_yr} Projected)</div>
            <div class="card-value">₹{latest_gddp:,.0f} Cr</div>
            <div class="card-sub">+{gddp_growth}% Projected Expansion</div>
        </div>
        <div class="card">
            <div class="card-label">Spatial Grid Resolution</div>
            <div class="card-value">{res_w} &times; {res_h} px</div>
            <div class="card-sub">{region_name} ({len(zones)} Tracked Zones)</div>
        </div>
        <div class="card">
            <div class="card-label">Satellite Temporal Depth</div>
            <div class="card-value">{len(images_hist)} Frames</div>
            <div class="card-sub">{start_yr}–{end_yr} Time-Series</div>
        </div>
    </div>

    <!-- Dual Comparison: Optical Ground Truth / Baseline vs Night Light AI -->
    <div class="comparison-layout">
        <div class="viewer-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h3 style="margin: 0; color: var(--accent-blue);">{left_title}</h3>
                <span class="badge badge-info">{left_badge}</span>
            </div>
            <p style="font-size: 12px; color: var(--text-muted); margin: 4px 0 10px 0;">{left_desc}</p>
            <img class="image-frame" src="data:image/jpeg;base64,{opt_b64}" alt="Optical Satellite with Landmarks">
            <p style="font-size: 11px; color: #8b949e; margin-top: 8px;">
                Anchors: <strong>Tumkur Core (Center)</strong> | <strong>Vasanthanarasapura (NW)</strong> | <strong>Dabaspet & Nelamangala (SE Corridor)</strong> | <strong>Kunigal (South)</strong> | <strong>Tiptur & Gubbi (West)</strong> | <strong>Sira (North)</strong>
            </p>
        </div>

        <!-- Nighttime Satellite AI Viewer -->
        <div class="viewer-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h3 style="margin: 0; color: var(--accent-orange);">Nighttime Radiance (NASA VIIRS + AI Forecast)</h3>
                <span id="typeTag" class="tag-pill tag-pred">AI PREDICTION</span>
            </div>
            <p style="font-size: 12px; color: var(--text-muted); margin: 4px 0 10px 0;">Scrub slider to watch 10 years of monthly economic development and future growth.</p>
            <img id="satelliteFrame" class="image-frame" src="data:image/jpeg;base64,{encoded_frames[-1]}" alt="Satellite Night Light">

            <div class="slider-control">
                <span class="year-display" id="yearLabel">{sampled_labels[-1]}</span>
                <input type="range" id="timeSlider" min="0" max="{len(sampled_labels) - 1}" value="{len(sampled_labels) - 1}" step="1" oninput="updateFrame(this.value)">
            </div>
            <p style="font-size: 11px; color: #8b949e; margin-top: 8px;">
                Brightness directly measures active factories, logistics, electrified commerce, and infrastructure expansion.
            </p>
        </div>
    </div>

    <!-- Chart Section -->
    <div class="card" style="margin-bottom: 24px;">
        <h3 style="margin: 0 0 12px 0; color: var(--accent-blue);">Nowcasted Economic Output: 10-Year Trajectory & AI Projections (GDDP in ₹ Crores)</h3>
        <canvas id="econChart" height="90"></canvas>
    </div>

    <!-- Opportunity Zones Table -->
    <div class="card">
        <h3 style="margin: 0 0 12px 0; color: var(--accent-blue);">Landmark Opportunity Matrix (Where Development is Expanding & Who Pays)</h3>
        <table>
            <thead>
                <tr>
                    <th>Landmark / Hub</th>
                    <th>Type</th>
                    <th>Growth Velocity</th>
                    <th>Economic Status</th>
                    <th>Strategic Action (Fintech / Infrastructure / Logistics)</th>
                </tr>
            </thead>
            <tbody>
                {zones_html}
            </tbody>
        </table>
    </div>

    <script>
        const labels = {json.dumps(sampled_labels)};
        const histCount = {len(sample_indices)};
        const frames = {json.dumps(encoded_frames)};

        function updateFrame(index) {{
            document.getElementById('yearLabel').innerText = labels[index];
            document.getElementById('satelliteFrame').src = "data:image/jpeg;base64," + frames[index];

            const tag = document.getElementById('typeTag');
            if (index < histCount) {{
                tag.innerText = "VIIRS SATELLITE";
                tag.className = "tag-pill tag-hist";
            }} else {{
                tag.innerText = "AI PREDICTION";
                tag.className = "tag-pill tag-pred";
            }}
        }}

        // Initialize Chart.js
        const ctx = document.getElementById('econChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(total_labels)},
                datasets: [
                    {{
                        label: 'Estimated GDDP (₹ Crores)',
                        data: {json.dumps(total_gddp)},
                        borderColor: '#3fb950',
                        backgroundColor: 'rgba(63, 185, 80, 0.1)',
                        fill: true,
                        tension: 0.25,
                        borderWidth: 2.5,
                        pointRadius: 1
                    }}
                ]
            }},
            options: {{
                responsive: true,
                scales: {{
                    y: {{
                        grid: {{ color: '#30363d' }},
                        ticks: {{ color: '#8b949e' }}
                    }},
                    x: {{
                        grid: {{ color: '#30363d' }},
                        ticks: {{ 
                            color: '#8b949e',
                            maxTicksLimit: 12
                        }}
                    }}
                }},
                plugins: {{
                    legend: {{ labels: {{ color: '#f0f6fc' }} }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"[Visualizer] Generated interactive dashboard at: {output_path}")


def generate_predictions_table_html(hist_df, pred_df, zones, output_path, region_name="Tumkur", dashboard_filename=None):
    """
    Generates a standalone tabular HTML report of AI economic forecasts,
    opportunity zone rankings, and actionable development suggestions.
    """
    start_yr = int(hist_df['year'].iloc[0])
    end_yr = int(hist_df['year'].iloc[-1])
    n_frames = len(hist_df)

    # Table 1: Economic Forecast Rows
    forecast_rows = ""
    for _, row in pred_df.iterrows():
        yr = int(row['year'])
        gddp = row['predicted_gddp_crores']
        low = row['confidence_interval_low_gddp']
        high = row['confidence_interval_high_gddp']
        eidx = row['economic_activity_index']
        sol = row['predicted_sum_of_lights']
        lit = row['predicted_lit_area_pct']
        forecast_rows += f"""
        <tr>
          <td><strong>{yr}</strong></td>
          <td class="num up">₹{gddp:,.0f} Cr</td>
          <td class="num muted">₹{low:,.0f} Cr</td>
          <td class="num muted">₹{high:,.0f} Cr</td>
          <td>
            <div style="display:flex;align-items:center;gap:8px;">
              <div class="bar-wrap"><div class="bar-fill" style="width:{min(100, eidx):.1f}%;background:#58a6ff;"></div></div>
              <span class="num">{eidx:.1f} / 100</span>
            </div>
          </td>
          <td class="num">{sol:,.0f}</td>
          <td class="num">{lit:.1f}%</td>
          <td><span class="badge badge-blue">Forecast</span></td>
        </tr>"""

    # Table 2 & 3: Zone Rankings + AI Suggestions
    max_growth = max([abs(z['growth_pct']) for z in zones], default=1.0) or 1.0
    zone_rows = ""
    suggestion_rows = ""
    for idx, z in enumerate(zones[:10], start=1):
        g = z['growth_pct']
        bar_w = min(100.0, max(3.0, (abs(g) / max_growth) * 100.0))
        if g > 20:
            g_cls, bar_col, b_cls, prio = "up", "#3fb950", "badge-green", ("badge-green", "High")
        elif g > 8:
            g_cls, bar_col, b_cls, prio = "flat", "#d29922", "badge-yellow", ("badge-blue", "Medium")
        else:
            g_cls, bar_col, b_cls, prio = "down", "#8b949e", "badge-yellow", ("badge-yellow", "Advisory")

        zone_rows += f"""
        <tr>
          <td class="num"><strong>#{idx}</strong></td>
          <td><strong>{z['name']}</strong></td>
          <td><span class="badge badge-purple">{z['type'].title()}</span></td>
          <td class="num">{z['current_flux']:,.0f}</td>
          <td class="num">{z['future_flux']:,.0f}</td>
          <td class="num {g_cls}">{g:+.1f}%</td>
          <td><div class="bar-wrap"><div class="bar-fill" style="width:{bar_w:.0f}%;background:{bar_col};"></div></div></td>
          <td><span class="badge {b_cls}">{z['status']}</span></td>
        </tr>"""

        suggestion_rows += f"""
        <tr>
          <td class="num">{idx}</td>
          <td><strong>{z['recommended_action']}</strong></td>
          <td>{z['name']}</td>
          <td class="num {g_cls}">{g:+.1f}% projected flux</td>
          <td><span class="badge {prio[0]}">{prio[1]}</span></td>
          <td>{z['status']}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{region_name} – AI Predictions & Suggestions</title>
<style>
  :root {{
    --bg: #0d1117; --card: #161b22; --border: #30363d;
    --accent: #58a6ff; --green: #3fb950; --yellow: #d29922;
    --red: #f85149; --text: #e6edf3; --muted: #8b949e;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; padding: 32px 24px; }}
  h1 {{ font-size: 1.6rem; font-weight: 700; color: var(--accent); margin-bottom: 4px; }}
  .subtitle {{ color: var(--muted); font-size: 0.85rem; margin-bottom: 32px; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; }}
  .badge-blue   {{ background: #1f3f6e; color: #58a6ff; }}
  .badge-green  {{ background: #1a3a2a; color: #3fb950; }}
  .badge-yellow {{ background: #3d2e00; color: #d29922; }}
  .badge-purple {{ background: #2d1f4e; color: #bc8cff; }}
  section {{ margin-bottom: 40px; }}
  h2 {{ font-size: 1.05rem; font-weight: 600; border-left: 3px solid var(--accent); padding-left: 10px; margin-bottom: 14px; }}
  .tbl-wrap {{ overflow-x: auto; border-radius: 10px; border: 1px solid var(--border); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; }}
  thead th {{ background: #1c2230; color: var(--muted); text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.6px; padding: 12px 16px; text-align: left; border-bottom: 1px solid var(--border); }}
  tbody tr {{ border-bottom: 1px solid #21272f; }}
  tbody tr:last-child {{ border-bottom: none; }}
  tbody tr:hover {{ background: #1c2230; }}
  tbody td {{ padding: 11px 16px; vertical-align: middle; }}
  .num {{ font-variant-numeric: tabular-nums; font-family: 'Cascadia Code', 'Consolas', monospace; }}
  .muted {{ color: var(--muted); }}
  .up {{ color: var(--green); font-weight: 600; }}
  .down {{ color: var(--red); font-weight: 600; }}
  .flat {{ color: var(--yellow); font-weight: 600; }}
  .bar-wrap {{ width: 110px; background: #21272f; border-radius: 4px; height: 7px; }}
  .bar-fill {{ height: 7px; border-radius: 4px; }}
  .top-bar {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; margin-bottom: 28px; flex-wrap: wrap; }}
  .top-bar .subtitle {{ margin-bottom: 0; }}
  .nav-btn {{
    display: inline-flex; align-items: center; gap: 6px;
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: #ffffff; text-decoration: none; font-weight: 600; font-size: 0.85rem;
    padding: 9px 16px; border-radius: 8px; border: 1px solid rgba(88, 166, 255, 0.5);
    box-shadow: 0 2px 10px rgba(31, 111, 235, 0.3); white-space: nowrap;
    transition: transform 0.15s, box-shadow 0.15s;
  }}
  .nav-btn:hover {{ transform: translateY(-1px); box-shadow: 0 4px 14px rgba(56, 139, 253, 0.45); }}
  footer {{ margin-top: 48px; color: var(--muted); font-size: 0.75rem; border-top: 1px solid var(--border); padding-top: 16px; }}
</style>
</head>
<body>
<div class="top-bar">
  <div>
    <h1>🛰️ {region_name} — AI Predictions &amp; Suggestions</h1>
    <p class="subtitle">Track A · Boeing Aeromania '26 · Auto-generated from {n_frames} satellite frames ({start_yr}–{end_yr})</p>
  </div>
  {f'<a href="{dashboard_filename}" class="nav-btn">&#128752;&#65039; Open Interactive Dashboard &rarr;</a>' if dashboard_filename else ''}
</div>

<section>
  <h2>📈 Economic Activity Forecast</h2>
  <div class="tbl-wrap">
    <table>
      <thead>
        <tr><th>Year</th><th>Predicted GDDP</th><th>Lower Bound</th><th>Upper Bound</th><th>Econ Index</th><th>Sum of Lights</th><th>Lit Area</th><th>Status</th></tr>
      </thead>
      <tbody>{forecast_rows}</tbody>
    </table>
  </div>
</section>

<section>
  <h2>🏭 Opportunity Zone Rankings</h2>
  <div class="tbl-wrap">
    <table>
      <thead>
        <tr><th>Rank</th><th>Zone / Sector</th><th>Type</th><th>Current Flux</th><th>Projected Flux</th><th>Growth</th><th>Velocity</th><th>Classification</th></tr>
      </thead>
      <tbody>{zone_rows}</tbody>
    </table>
  </div>
</section>

<section>
  <h2>💡 AI Suggestions — Development Priorities</h2>
  <div class="tbl-wrap">
    <table>
      <thead>
        <tr><th>#</th><th>Recommended Action</th><th>Target Zone</th><th>Signal</th><th>Priority</th><th>Zone Status</th></tr>
      </thead>
      <tbody>{suggestion_rows}</tbody>
    </table>
  </div>
</section>

<footer>Generated by Spatio-Temporal Night-Light AI · Region: {region_name} · {n_frames} frames ({start_yr}–{end_yr})</footer>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"[Visualizer] Generated predictions & suggestions table at: {output_path}")
