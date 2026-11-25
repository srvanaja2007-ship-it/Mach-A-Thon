import os
import cv2
import json
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from download_online_dataset import LANDMARKS, coord_to_pixel

def render_optical_vs_nightlights_comparison(optical_path, ntl_img, output_path):
    """
    Renders side-by-side comparison between daytime optical satellite imagery (with landmarks)
    and nighttime satellite radiance (VIIRS).
    """
    opt_bgr = cv2.imread(optical_path)
    if opt_bgr is None:
        return
    opt_rgb = cv2.cvtColor(opt_bgr, cv2.COLOR_BGR2RGB)

    ntl_colored = cv2.applyColorMap(np.clip(ntl_img, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    ntl_rgb = cv2.cvtColor(ntl_colored, cv2.COLOR_BGR2RGB)

    # Draw landmarks on night light image too
    for lm in LANDMARKS:
        px, py = coord_to_pixel(lm['lat'], lm['lon'], w=ntl_img.shape[1], h=ntl_img.shape[0])
        cv2.circle(ntl_rgb, (px, py), 5, (0, 255, 255), -1)
        cv2.putText(ntl_rgb, lm['name'][:14], (px + 6, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='#0d1117')
    ax1.imshow(opt_rgb)
    ax1.set_title("Daytime Optical Satellite (High-Res + Landmarks)", color='#58a6ff', fontsize=14, fontweight='bold', pad=10)
    ax1.axis('off')

    ax2.imshow(ntl_rgb)
    ax2.set_title("Nighttime Satellite Radiance (NASA VIIRS - Economic Activity)", color='#f0883e', fontsize=14, fontweight='bold', pad=10)
    ax2.axis('off')

    plt.suptitle("Tumkur District & Outskirts: Optical Ground Truth vs Satellite Night Lights (9x Area)", 
                 color='#f0f6fc', fontsize=16, fontweight='bold', y=0.96)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[Visualizer] Saved Optical vs Nightlights comparison to: {output_path}")

def render_plot_timeline(images_hist, years_hist, images_future, years_future, output_path):
    """
    Renders multi-panel timeline comparing key historical years with future AI predictions.
    """
    # Select sample representative years across the 10-year span
    key_years = [2015, 2017, 2019, 2021, 2023, 2024]
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

            # Draw landmark anchors
            for lm in LANDMARKS[:5]:
                px, py = coord_to_pixel(lm['lat'], lm['lon'], w=img.shape[1], h=img.shape[0])
                ax.plot(px, py, marker='o', markersize=4, color='#00ffcc')
                if idx == total - 1 or idx == 0:
                    ax.text(px + 6, py - 4, lm['name'][:12], color='#00ffcc', fontsize=7, fontweight='bold')
        else:
            ax.axis('off')

    plt.suptitle("Tumkur Spatio-Temporal Development: 10-Year Satellite History & AI Forecast (9x Area)", 
                 color='#f0f6fc', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[Visualizer] Saved timeline progression to: {output_path}")

def render_plot_metrics(hist_df, pred_df, zones, output_path):
    """
    Renders 4 analytical plots showing monthly SOL, GDDP forecast, landmark growth, and economic velocity.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 11), facecolor='#0d1117')
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
        ax.grid(True, linestyle='--', alpha=0.3, color='#8b949e')

    # 1. 10-Year Monthly SOL & 12-Month Smooth Trend
    t_hist = hist_df['timestamp'].values
    ax1.plot(t_hist, hist_df['sum_of_lights'], color='#30363d', alpha=0.6, label='Monthly Raw Flux')
    ax1.plot(t_hist, hist_df['sol_trend_smooth'], color='#58a6ff', linewidth=2.5, label='12-Mo Seasonally Adjusted')
    # Connect to prediction
    last_t = t_hist[-1]
    last_sol = hist_df['sol_trend_smooth'].iloc[-1]
    pred_t = [last_t] + list(pred_df['year'].values)
    pred_sol = [last_sol] + list(pred_df['predicted_sum_of_lights'].values)
    ax1.plot(pred_t, pred_sol, color='#f0883e', linestyle='--', linewidth=2.5, label='AI Forecast Trend')
    ax1.set_title("Radiant Flux (10-Year Monthly VIIRS + Forecast)", color='#f0f6fc', fontsize=12, fontweight='bold')
    ax1.set_ylabel("Sum of Lights Flux", color='#8b949e')
    ax1.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#f0f6fc')

    # 2. Estimated Tumkur GDDP (₹ Crores)
    base_sol = hist_df['sol_trend_smooth'].iloc[0]
    hist_gddp = 32000.0 * ((hist_df['sol_trend_smooth'] / base_sol) ** 0.75)
    ax2.plot(t_hist, hist_gddp, color='#3fb950', linewidth=2.5, label='Nowcasted GDDP (₹ Cr)')
    pred_gddp = [hist_gddp.iloc[-1]] + list(pred_df['predicted_gddp_crores'].values)
    ax2.plot(pred_t, pred_gddp, color='#d29922', linestyle='--', linewidth=2.5, label='AI Projected GDDP')
    
    ci_low = [hist_gddp.iloc[-1]] + list(pred_df['confidence_interval_low_gddp'].values)
    ci_high = [hist_gddp.iloc[-1]] + list(pred_df['confidence_interval_high_gddp'].values)
    ax2.fill_between(pred_t, ci_low, ci_high, color='#d29922', alpha=0.2, label='95% Confidence Band')
    ax2.set_title("Estimated Tumkur Output / GDDP (₹ Crores)", color='#f0f6fc', fontsize=12, fontweight='bold')
    ax2.set_ylabel("₹ in Crores", color='#8b949e')
    ax2.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#f0f6fc')

    # 3. Landmark Economic Growth Velocity (% Increase)
    top_zones = zones[:6]
    names = [z['name'][:18] for z in top_zones][::-1]
    growths = [z['growth_pct'] for z in top_zones][::-1]
    colors = ['#3fb950' if g > 25 else '#58a6ff' for g in growths]
    ax3.barh(names, growths, color=colors, height=0.6)
    ax3.set_title("Projected Growth Velocity by Landmark Hub (%)", color='#f0f6fc', fontsize=12, fontweight='bold')
    ax3.set_xlabel("% Projected Radiance Expansion", color='#8b949e')

    # 4. Economic Activity Index
    hist_idx = (hist_gddp / 32000.0) * 35.0
    pred_idx = [hist_idx.iloc[-1]] + list(pred_df['economic_activity_index'].values)
    ax4.plot(t_hist, hist_idx, color='#bc8cff', linewidth=2.5, label='Historical Econ Index')
    ax4.plot(pred_t, pred_idx, color='#f778ba', linestyle='--', linewidth=2.5, label='Projected Index')
    ax4.set_title("District Economic Activity Index (0 - 100)", color='#f0f6fc', fontsize=12, fontweight='bold')
    ax4.set_ylabel("Index Score", color='#8b949e')
    ax4.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#f0f6fc')

    plt.suptitle("Tumkur Economic Nowcasting & Growth Projections (10-Year Satellite Model)", color='#f0f6fc', fontsize=16, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[Visualizer] Saved metrics forecast plot to: {output_path}")

def generate_interactive_html_dashboard(hist_df, pred_df, images_hist, timestamps_hist, images_future, years_future, zones, optical_path, output_path):
    """
    Creates standalone interactive HTML dashboard featuring:
    - Side-by-side Optical Satellite View (with landmarks) vs Night Lights
    - 120-Month Scrubbable Time Slider
    - Live GDDP Nowcasting & Opportunity Zones table
    """
    # Sample 24 representative monthly frames + all future frames to keep HTML size under 6 MB
    sample_indices = np.linspace(0, len(images_hist) - 1, 24, dtype=int)
    sampled_imgs = [images_hist[i] for i in sample_indices]
    sampled_timestamps = [timestamps_hist[i] for i in sample_indices]
    sampled_labels = [f"{int(timestamps_hist[i])}-{int((timestamps_hist[i] % 1) * 12) + 1:02d}" for i in sample_indices]

    for yr, fut_img in images_future:
        sampled_imgs.append(fut_img)
        sampled_timestamps.append(float(yr))
        sampled_labels.append(f"{yr} (AI PREDICTED)")

    encoded_frames = []
    for img in sampled_imgs:
        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(img_u8, cv2.COLORMAP_INFERNO)
        _, buf = cv2.imencode('.jpg', colored, [cv2.IMWRITE_JPEG_QUALITY, 80])
        encoded_frames.append(base64.b64encode(buf).decode('utf-8'))

    # Encode optical image with landmarks
    opt_bgr = cv2.imread(optical_path)
    if opt_bgr is not None:
        _, opt_buf = cv2.imencode('.jpg', opt_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        opt_b64 = base64.b64encode(opt_buf).decode('utf-8')
    else:
        opt_b64 = encoded_frames[0]

    # Chart data
    chart_years = [f"{int(t)}" if (t % 1) < 0.1 else "" for t in hist_df['timestamp']]
    chart_years_clean = [f"{int(t)}-{int((t%1)*12)+1:02d}" for t in hist_df['timestamp']]
    base_sol = hist_df['sol_trend_smooth'].iloc[0]
    hist_gddp = list(np.round(32000.0 * ((hist_df['sol_trend_smooth'] / base_sol) ** 0.75), 1))

    # Add predicted points
    future_labels_chart = [str(yr) for yr in years_future]
    future_gddp_chart = list(np.round(pred_df['predicted_gddp_crores'], 1))

    total_labels = chart_years_clean + future_labels_chart
    total_gddp = hist_gddp + future_gddp_chart

    latest_gddp = total_gddp[-1]
    baseline_gddp = total_gddp[0]
    gddp_growth = round(((latest_gddp - baseline_gddp) / baseline_gddp) * 100.0, 1)

    zones_html = ""
    for z in zones[:7]:
        color_class = "badge-success" if z['growth_pct'] > 25 else "badge-warning"
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
    </style>
</head>
<body>

    <div class="header">
        <div>
            <h1>Orbit to Economy: Tumkur & Outskirts (9x Area Spatio-Temporal AI)</h1>
            <p>10-Year Satellite Night Lights (NASA VIIRS Monthly) vs High-Res Optical Ground Truth & 2027 Projections</p>
        </div>
        <div class="badge-track">Track A: Read the Planet &bull; Boeing Aeromania '26</div>
    </div>

    <div class="grid">
        <div class="card">
            <div class="card-label">Baseline GDDP (2015)</div>
            <div class="card-value">₹32,000 Cr</div>
            <div class="card-sub">State Survey Reference</div>
        </div>
        <div class="card">
            <div class="card-label">Nowcasted Output (2027 Projected)</div>
            <div class="card-value">₹{latest_gddp:,.0f} Cr</div>
            <div class="card-sub">+{gddp_growth}% 10-Yr Expansion</div>
        </div>
        <div class="card">
            <div class="card-label">Spatial Coverage (9x Area)</div>
            <div class="card-value">13,200 km²</div>
            <div class="card-sub">Tumkur Core + 9 Outskirt Hubs</div>
        </div>
        <div class="card">
            <div class="card-label">Monthly Satellite Depth</div>
            <div class="card-value">120 Months</div>
            <div class="card-sub">2015–2024 Time-Series</div>
        </div>
    </div>

    <!-- Dual Comparison: Optical Ground Truth vs Night Light AI -->
    <div class="comparison-layout">
        <!-- Optical Satellite View with Landmarks -->
        <div class="viewer-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h3 style="margin: 0; color: var(--accent-blue);">Optical Daytime Satellite (Landmarks Ground Truth)</h3>
                <span class="badge badge-info">ESRI HIGH-RES SATELLITE</span>
            </div>
            <p style="font-size: 12px; color: var(--text-muted); margin: 4px 0 10px 0;">Ground truth showing road networks, terrain, Vasanthanarasapura, NH-48 & outskirt towns.</p>
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
    print(f"[Visualizer] Generated 10-Year interactive dashboard at: {output_path}")
