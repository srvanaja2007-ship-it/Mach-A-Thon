import glob, cv2, numpy as np, json

files = sorted(glob.glob('Eco_Act/Dataset/monthly_10yr/*.png'))
stats = []
for f in files:
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    name = f.replace('\\\\', '/').replace('\\', '/').split('/')[-1]
    stats.append({
        'file': name,
        'mean': float(img.mean()),
        'std': float(img.std()),
        'min': int(img.min()),
        'max': int(img.max()),
        'p5': float(np.percentile(img, 5)),
        'p15': float(np.percentile(img, 15)),
        'p50': float(np.percentile(img, 50)),
        'p95': float(np.percentile(img, 95)),
        'dark_pct': float((img < 30).sum() / img.size * 100),
        'saturated_pct': float((img > 240).sum() / img.size * 100)
    })

means = [s['mean'] for s in stats]
print(f'Mean range: min={min(means):.1f}  max={max(means):.1f}  ratio={max(means)/max(0.001,min(means)):.1f}x')

print('\nTOP 15 BRIGHT (cloud-saturated / moonlit):')
for s in sorted(stats, key=lambda x: x['mean'], reverse=True)[:15]:
    print(f"  {s['file']}: mean={s['mean']:.1f} std={s['std']:.1f} sat%={s['saturated_pct']:.1f}% dark%={s['dark_pct']:.1f}%")

print('\nTOP 15 DARK (sensor gaps / poor coverage):')
for s in sorted(stats, key=lambda x: x['mean'])[:15]:
    print(f"  {s['file']}: mean={s['mean']:.1f} std={s['std']:.1f} dark%={s['dark_pct']:.1f}%")

print('\nSAVING stats json...')
with open('Eco_Act/Dataset/image_stats.json', 'w') as fp:
    json.dump(stats, fp, indent=2)
print('Done.')
