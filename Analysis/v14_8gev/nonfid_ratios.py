import numpy as np
import uproot
import glob
import re
import math
from tqdm import tqdm
import awkward
import concurrent.futures
executor = concurrent.futures.ThreadPoolExecutor(20)

SP_TARGET_DOWN_Z = 0.1767
ECAL_SP_Z = 239.9985
ECAL_FACE_Z = 247.932
CELL_RADIUS = 5.0

def dist(p1, p2):
    return math.sqrt(np.sum( ( np.array(p1) - np.array(p2) )**2 ))

def projection(Recoilx, Recoily, Recoilz, RPx, RPy, RPz, HitZ):
    x_final = Recoilx + RPx/RPz*(HitZ - Recoilz) if RPz else 0
    y_final = Recoily + RPy/RPz*(HitZ - Recoilz) if RPy else 0
    return (x_final, y_final)

def _load_cellMap(version='v13'):
    cellMap = {}
    for i, x, y in np.loadtxt('/home/duncansw/LDMX-scripts/GraphNet/data/%s/cellmodule.txt' % version):
        cellMap[i] = (x, y)
    global cells 
    cells = np.array(list(cellMap.values()))
    print("Loaded {} detector info".format(version))

def pad_array(arr):
    arr = awkward.pad_none(arr, 1, clip=True)
    arr = awkward.fill_none(arr, 0)
    return awkward.flatten(arr)

file_templates = {
    0.001:  '/home/vamitamas/Samples8GeV/Ap0.001GeV_sim/*.root',
    0.01:  '/home/vamitamas/Samples8GeV/Ap0.01GeV_sim/*.root',
    0.1:   '/home/vamitamas/Samples8GeV/Ap0.1GeV_sim/*.root',
    1.0:   '/home/vamitamas/Samples8GeV/Ap1GeV_sim/*.root',
    0:     '/home/vamitamas/Samples8GeV/v3.3.3_ecalPN*/*.root'
}

_load_cellMap()

nonfid_ratios = {}
for mass in file_templates.keys():
    
    print(f"==== m = {mass} ====", flush=True)
    
    # different branch name syntax for signal vs. bkg
    if mass:
        branchList = ['TargetScoringPlaneHits_signal/TargetScoringPlaneHits_signal.pdgID_', 'TargetScoringPlaneHits_signal/TargetScoringPlaneHits_signal.x_',
                      'TargetScoringPlaneHits_signal/TargetScoringPlaneHits_signal.y_', 'TargetScoringPlaneHits_signal/TargetScoringPlaneHits_signal.z_',
                     'TargetScoringPlaneHits_signal/TargetScoringPlaneHits_signal.px_', 'TargetScoringPlaneHits_signal/TargetScoringPlaneHits_signal.py_',
                     'TargetScoringPlaneHits_signal/TargetScoringPlaneHits_signal.pz_', 'TriggerSums20Layers_signal/pass_'] 
    
    else:
        branchList = ['TargetScoringPlaneHits_sim/TargetScoringPlaneHits_sim.pdgID_', 'TargetScoringPlaneHits_sim/TargetScoringPlaneHits_sim.x_',
                      'TargetScoringPlaneHits_sim/TargetScoringPlaneHits_sim.y_', 'TargetScoringPlaneHits_sim/TargetScoringPlaneHits_sim.z_',
                     'TargetScoringPlaneHits_sim/TargetScoringPlaneHits_sim.px_', 'TargetScoringPlaneHits_sim/TargetScoringPlaneHits_sim.py_',
                     'TargetScoringPlaneHits_sim/TargetScoringPlaneHits_sim.pz_'] 
    
    file_list = glob.glob(file_templates[mass])
    nFiles = len(file_list)
    
    nEvents = 0
    nNonFid = 0
    for i, filename in tqdm(enumerate(file_list), total=nFiles):
        if nEvents>=1e4:
            break
        #if i == 1:
            #break
        try:
            with uproot.open(filename, interpretation_executor=executor) as file:
                if not file.keys(): # if no keys in file
                    print(f"FOUND ZOMBIE: {filename}  SKIPPING...", flush=True)
                    continue
                    
            with uproot.open(filename, interpretation_executor=executor)['LDMX_Events'] as t:
                if not t.keys(): # if no keys in 'LDMX_Events'
                    print(f"FOUND ZOMBIE: {filename}  SKIPPING...", flush=True)
                    continue
                key_miss = False
                for branch in branchList:
                    if not re.split('/', branch)[0] in t.keys(): # if one or more desired keys missing
                        key_miss = True
                        break
                if key_miss:
                    print(f"MISSING KEY(S) IN: {filename}  SKIPPING...", flush=True)
                    continue
                data = t.arrays(branchList, interpretation_executor=executor)

                # tsp leaves
                pdgID = data[branchList[0]]
                x = data[branchList[1]]
                y = data[branchList[2]]
                z = data[branchList[3]]
                px = data[branchList[4]]
                py = data[branchList[5]]
                pz = data[branchList[6]]
                
                print(len(z))
                
                # Apply trigger for signal
                if mass:
                    tskimmed_data = {}
                    trig_pass = data[branchList[7]]
                    for branch in branchList:
                        tskimmed_data[branch] = data[branch][trig_pass]
                    pdgID = tskimmed_data[branchList[0]]
                    x = tskimmed_data[branchList[1]]
                    y = tskimmed_data[branchList[2]]
                    z = tskimmed_data[branchList[3]]
                    px = tskimmed_data[branchList[4]]
                    py = tskimmed_data[branchList[5]]
                    pz = tskimmed_data[branchList[6]]
                        
                print(len(z))

                nEvents += len(z)

                # Select recoil electron at downstream tsp (maximal forward moving pz electron)
                e_cut = []
                for i in range(len(px)):
                    e_cut.append([])
                    for j in range(len(px[i])):
                        e_cut[i].append(False)
                for i in range(len(px)):
                    maxPz = 0
                    e_index = 0
                    for j in range(len(px[i])):
                        #P = np.sqrt(px[i][j]**2 + py[i][j]**2 + pz[i][j]**2)
                        if (pdgID[i][j] == 11 and z[i][j] > 0.17 and z[i][j] < 0.18 and pz[i][j] > maxPz):
                            maxPz = pz[i][j]
                            e_index = j
                    if maxPz > 0:
                        e_cut[i][e_index] = True

                recoilX = pad_array(x[e_cut])
                recoilY = pad_array(y[e_cut])
                recoilPx = pad_array(px[e_cut])
                recoilPy = pad_array(py[e_cut])
                recoilPz = pad_array(pz[e_cut])

                # fiducial cut
                N = len(recoilX)
                f_cut = np.zeros(N, dtype = bool)
                for i in range(N):
                    fiducial = False
                    fXY = projection(recoilX[i], recoilY[i], SP_TARGET_DOWN_Z, recoilPx[i], recoilPy[i], recoilPz[i], ECAL_FACE_Z)
                    if not (recoilX[i] == 0 and recoilY[i] == 0 and recoilPx[i] == 0 and recoilPy[i] == 0 and recoilPz[i] == 0):
                        for j in range(len(cells)):
                            celldis = dist(cells[j], fXY)
                            if celldis <= CELL_RADIUS:
                                fiducial = True
                                break
                    if fiducial == True:
                        f_cut[i] = 1

                nNonFid += len(recoilX[f_cut==0])
                
        except OSError:
            continue
            
        
    nonfid_ratios[mass] = nNonFid / nEvents if nEvents else 'no events'
    
print(nonfid_ratios)