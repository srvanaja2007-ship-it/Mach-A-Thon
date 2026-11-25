import sys, pathlib
proj_root = pathlib.Path(r'C:\Users\T.ASHOK\Desktop\partb_2\Mach-A-Thon')
sys.path.append(str(proj_root))
from LEO_Sim.onboard_processor.processor import run
img = r'C:\Users\T.ASHOK\Desktop\partb_2\Mach-A-Thon\Eco_Act\Dataset\tumkur_reference_viirs\tumkur_viirs_2021.png'
print('Running processor...')
res = run(img)
print('Result dict:', res)
