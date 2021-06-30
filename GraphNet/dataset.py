from __future__ import print_function

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import glob
import tqdm
import uproot
import awkward
import concurrent.futures
import math
import psutil
import gc  # May reduce RAM usage

executor = concurrent.futures.ThreadPoolExecutor(12)

torch.set_default_dtype(torch.float32)

#ecalBranches = [  # EcalVeto data to save.  Could add more, but probably unnecessary.
#    'discValue_',
#    'recoilX_',
#    'recoilY_',
#    ]

MAX_NUM_ECAL_HITS = 110  # Still need this to determine array sizes...110 gives >98% efficiency, so using that.
MAX_ISO_ENERGY = 650  # 600 gives slightly under 95% sig eff; shoot for a bit over

# NEW:  LayerZ data (may be outdated)
# Assumed outdated; not currently used

# NEW: Radius of containment data
#from 2e (currently not used)
radius_beam_68 = [4.73798004, 4.80501156, 4.77108164, 4.53839401, 4.73273021,
4.76662872, 5.76994967, 5.92028271, 7.28770932, 7.60723209,
9.36050277, 10.03247442, 12.14656399, 13.16076587, 15.88429816,
17.03559932, 20.32607264, 21.75096888, 24.98745754, 27.02031225,
30.78043038, 33.03033267, 37.55088662, 40.14062264, 47.95964745,
55.96441035, 66.33128366, 70.42649416, 86.68563278, 102.49022815,
119.06854141, 121.20048803, 127.5236134, 121.99024095]

#from 1e
radius_recoil_68_p_0_500_theta_0_10 = [4.045666158618167, 4.086393662224346, 4.359141107602775, 4.666549994726691, 5.8569181911416015, 6.559716356124256, 8.686967529043072, 10.063482736354674, 13.053528344041274, 14.883496407943747, 18.246694748611368, 19.939799900443724, 22.984795944506224, 25.14745829663406, 28.329169392203216, 29.468032123356345, 34.03271241527079, 35.03747443690781, 38.50748727211848, 39.41576583301171, 42.63622296033334, 45.41123601592071, 48.618139095742876, 48.11801717451056, 53.220539860213655, 58.87753380915155, 66.31550881539764, 72.94685877928593, 85.95506228335348, 89.20607201266672, 93.34370253818409, 96.59471226749734, 100.7323427930147, 103.98335252232795]

radius_recoil_68_p_500_1500_theta_0_10 = [4.081926458777424, 4.099431732299409, 4.262428482867968, 4.362017581473145, 4.831341579961153, 4.998346041276382, 6.2633736512415705, 6.588371889265881, 8.359969947444522, 9.015085558044309, 11.262722588206483, 12.250305471269183, 15.00547660437276, 16.187264014640103, 19.573764900578503, 20.68072032434797, 24.13797140783321, 25.62942209291236, 29.027596514735617, 30.215039667389316, 33.929540248019585, 36.12911729771914, 39.184563500620946, 42.02062468386282, 46.972125628650204, 47.78214816041894, 55.88428562462974, 59.15520134927332, 63.31816666637158, 66.58908239101515, 70.75204770811342, 74.022963432757, 78.18592874985525, 81.45684447449884]

radius_recoil_68_theta_10_20 = [4.0251896715647115, 4.071661598616328, 4.357690094817289, 4.760224640141712, 6.002480766325418, 6.667318981016246, 8.652513285172342, 9.72379373302137, 12.479492693251478, 14.058548828317289, 17.544872909347912, 19.43616066939176, 23.594162859513734, 25.197329065282954, 29.55995803074302, 31.768946746958296, 35.79247330197688, 37.27810357669942, 41.657281051476545, 42.628141392692626, 47.94208483539388, 49.9289473559796, 54.604030254423975, 53.958762417361655, 53.03339560920388, 57.026277390001425, 62.10810455035879, 66.10098633115634, 71.1828134915137, 75.17569527231124, 80.25752243266861, 84.25040421346615, 89.33223137382352, 93.32511315462106]

radius_recoil_68_theta_20_end = [4.0754238481177705, 4.193693485630508, 5.14209420056253, 6.114996249971468, 7.7376807326481645, 8.551663213602291, 11.129110612057813, 13.106293737495639, 17.186617323282082, 19.970887612094604, 25.04088272634407, 28.853696411302344, 34.72538105333071, 40.21218694947545, 46.07344239520299, 50.074953583805346, 62.944045771758645, 61.145621459396814, 69.86940198299047, 74.82378572939959, 89.4528387422834, 93.18228303096758, 92.51751129204555, 98.80228884380018, 111.17537347472128, 120.89712563907408, 133.27021026999518, 142.99196243434795, 155.36504706526904, 165.08679922962185, 177.45988386054293, 187.18163602489574, 199.55472065581682, 209.2764728201696]

radius_68 = [radius_beam_68,radius_recoil_68_p_0_500_theta_0_10, radius_recoil_68_p_500_1500_theta_0_10,radius_recoil_68_theta_10_20,radius_recoil_68_theta_20_end]

### DEFINING FUNCTIONS AND VARIABLES ###

scoringPlaneZ = 240.5015
ecalFaceZ = 248.35
cell_radius = 5.0


def projection(Recoilx, Recoily, Recoilz, RPx, RPy, RPz, HitZ):
    x_final = Recoilx + RPx/RPz*(HitZ - Recoilz)
    y_final = Recoily + RPy/RPz*(HitZ - Recoilz)
    return (x_final, y_final)
  
def dist(p1, p2):
    return math.sqrt(np.sum( ( np.array(p1) - np.array(p2) )**2 ))
  
def _concat(arrays, axis=0):
    if len(arrays) == 0:
        return np.array()
    if isinstance(arrays[0], np.ndarray):
        return np.concatenate(arrays, axis=axis)
    else:
        return awkward.concatenate(arrays, axis=axis)


class ECalHitsDataset(Dataset):

    def __init__(self, siglist, bkglist, load_range=(0, 1), apply_preselection=False, ignore_evt_limits=False, obs_branches=[], veto_branches=[], coord_ref=None, detector_version='v12'):
        super(ECalHitsDataset, self).__init__()

        # first load cell map
        self._load_cellMap(version=detector_version)
        self._id_branch = 'EcalRecHits_v12.id_'  # Technically not necessary anymore
        self._energy_branch = 'EcalRecHits_v12.energy_'
        #if veto_branches:
        ecal_veto_branches = ['EcalVeto_v12.'+b for b in veto_branches + ['summedTightIso_', 'discValue_']]
        #else:
        #    ecal_veto_branches = ['EcalVeto_v12.'+b for b in ['summedTightIso_', 'discValue_']]
        #self._test_branch = 'EcalVeto_v12'
        #self._x_branch = 'EcalRecHits_v12.xpos_'
        #self._y_branch = 'EcalRecHits_v12.ypos_'
        #self._z_branch = 'EcalRecHits_v12.zpos_'
        assert(detector_version == 'v12')
        #if detector_version == 'v9':
        #    print("WARNING:  Using v9 detector!  Case is not currently handled; will produce an error.")
        #    self._id_branch = 'ecalDigis_recon.id_'
        #    self._energy_branch = 'ecalDigis_recon.energy_'
        #if detector_version == 'v12':
        #    self._id_branch = 'EcalRecHits_v12.id_'
        #    self._energy_branch = 'EcalRecHits_v12.energy_'

        #self._branches = [self._id_branch, self._energy_branch, self._x_branch, self._y_branch, self._z_branch]
        self._branches = [self._id_branch, self._energy_branch] #, self._test_branch]

        self.extra_labels = []
        self.presel_eff = {}
        self.var_data = {}
        self.obs_data = {k:[] for k in obs_branches + ecal_veto_branches}
        
        ### VARIABLES FOR MAXPZ ###
        self.el_ = [] 
        print("self.el_ type: " + str(type(self.el_)))
        print("obs_branches:", obs_branches)
        print("ecal_veto_branches:", ecal_veto_branches)

        print('Using coord_ref=%s' % coord_ref)
        def _load_coord_ref(t, table):
            #print("    Usage before coord ref: {}".format(psutil.virtual_memory().percent))
            #print("***Finding recoil electrons!")
            # Find recoil electron (approx)
            # NOTE:  Requires precise knowledge of detector scoring plane!  Currently seems to be 240.5mm...(was plane 1)
            #        https://github.com/LDMX-Software/ldmx-sw/blob/master/Detectors/data/ldmx-det-v12/scoring_planes.gdml#L87-L88
            # NEW:  Also, ensure that the hit selected has max p.  Create array of max pz for each hit:
            #pz = t['EcalScoringPlaneHits_v12.pz_'].array()
            #pz_max = np.amax(pz, axis=1)
            #pz_max_ = awkward.from_iter([np.repeat(pz_max[i], len(pz[i])) for i in range(len(pz_max))])
            # Array is now [[pmax1, pmax1, ...],  [pmax2, pmax2, ...], ...]

            print("Usage before maxPz implementation: {}".format(psutil.virtual_memory().percent))

            trackid_ = t['EcalScoringPlaneHits_v12.trackID_'].array()
            id_ = t['EcalScoringPlaneHits_v12.pdgID_'].array()
            z_ = t['EcalScoringPlaneHits_v12.z_'].array()
            pz_ = t['EcalScoringPlaneHits_v12.pz_'].array()
            self.el_ = []
            has_e = []  # Also, need array to keep track of events w/ found SP e-.  If not found, False.
            print("Right after has_e...self.el_ type: " + str(type(self.el_)))
            for i in range(len(pz_)):
                pmax = 0  # Max pz for event i
                max_index = 0
                # print("Right before the append... self.el_ type: " + str(type(self.el_)))
                self.el_.append([])
                for j in range(len(pz_[i])):
                    if trackid_[i][j] == 1 and id_[i][j] == 11 and z_[i][j] > 240.0 and z_[i][j] < 241.001 and pz_[i][j] > pmax:
                            pmax = pz_[i][j]
                            max_index = j
                has_e.append(pmax != 0)
                for j in range(len(pz_[i])):
                    # If pz of hit = highest pz of all SP e- hits in event i, set mask to 1; else 0
                    self.el_[i].append(pz_[i][j] == pmax)
                if not has_e[i] and sum(self.el_[i]) == 0:  # Just make an arbitrary hit the SP hit; willb e handled by has_e later
                    self.el_[i][0] = True
                #print("1:", sum(el_[i]), ", ", len(el_[i]))
            self.el_ = awkward.from_iter(self.el_)

            #print("***Recoil electrons found!")
            #print(awkward.type(pz_))
            #print(awkward.type(el_))
            
            el = (t['EcalScoringPlaneHits_v12.pdgID_'].array() == 11) * \
                 (t['EcalScoringPlaneHits_v12.z_'].array() > 240.0) * \
                 (t['EcalScoringPlaneHits_v12.z_'].array() < 241.001) * \
                 (t['EcalScoringPlaneHits_v12.trackID_'].array() == 1) * \
                 self.el_

            print("Usage after maxPz implementation: {}".format(psutil.virtual_memory().percent))

            del id_
            del z_
            del pz_
            gc.collect()
            

            # Note:  pad() below ensures that only one SP electron is used if there's multiple (I believe)
            # pad() for awkward arrays is outdated; have to replace it...
            etraj_branches = ['EcalScoringPlaneHits_v12.x_', 'EcalScoringPlaneHits_v12.y_', 'EcalScoringPlaneHits_v12.z_',
                              'EcalScoringPlaneHits_v12.px_', 'EcalScoringPlaneHits_v12.py_', 'EcalScoringPlaneHits_v12.pz_']
            def _pad_array(arr):
                arr = awkward.pad_none(arr, 1, clip=True)
                arr = awkward.fill_none(arr, 0)
                return np.array(awkward.flatten(arr))  #NEW:  Include np conversion to allow stacking

            etraj_x_sp = _pad_array(t['EcalScoringPlaneHits_v12.x_'].array()[el])  #Arr of floats.  [0][0] fails.
            etraj_y_sp = _pad_array(t['EcalScoringPlaneHits_v12.y_'].array()[el])
            etraj_z_sp = _pad_array(t['EcalScoringPlaneHits_v12.z_'].array()[el])
            etraj_px_sp = _pad_array(t['EcalScoringPlaneHits_v12.px_'].array()[el])
            etraj_py_sp = _pad_array(t['EcalScoringPlaneHits_v12.py_'].array()[el])
            etraj_pz_sp = _pad_array(t['EcalScoringPlaneHits_v12.pz_'].array()[el])

            # Want [(x, y, z), ()...]
            #print(awkward.type(etraj_x_sp))
            #print(awkward.type(etraj_y_sp))
            etraj_sp = np.column_stack((etraj_x_sp, etraj_y_sp, etraj_z_sp))

            # Create vectors holding the electron/photon momenta so the trajectory projections can be found later
            # Set xtraj_p_norm relative to z=1 to make projecting easier:
            E_beam = 4000.0  # In GeV
            target_dist = 241.5 # distance from ecal to target, mm
            """
            etraj_p_norm = []
            for i in range(len(etraj_pz_sp)):
                if etraj_pz_sp[i] != 0 and has_e[i]:
                    etraj_p_norm.append((etraj_px_sp[i]/etraj_pz_sp[i], etraj_py_sp[i]/etraj_pz_sp[i], 1.0))
                else:
                    etraj_p_norm.append((0,0,0))
            """
            
            etraj_p_norm = np.zeros((len(etraj_pz_sp), 3), dtype='float32') #[]
            ptraj_p_norm = np.zeros((len(etraj_pz_sp), 3), dtype='float32') #[]
            ptraj_sp     = np.zeros((len(etraj_pz_sp), 3), dtype='float32') #[]  # (x, y, z) of projected photon hit @ ecal SP
            for i in range(len(etraj_pz_sp)):
                #print(ptraj_sp.shape)
                if etraj_pz_sp[i] != 0 and has_e[i]:
                    etraj_p_norm[i,:] = (etraj_px_sp[i]/etraj_pz_sp[i], etraj_py_sp[i]/etraj_pz_sp[i], 1.0)
                    ptraj_p_norm[i,:] = (-etraj_px_sp[i]/(E_beam - etraj_pz_sp[i]), -etraj_py_sp[i]/(E_beam - etraj_pz_sp[i]), 1.0)
                    #print(ptraj_sp.shape)
                    #print(ptraj_sp[i,:])
                    ptraj_sp[i,:]     = (etraj_x_sp[i] + target_dist*(ptraj_p_norm[i][0] - etraj_p_norm[i][0]),
                                         etraj_y_sp[i] + target_dist*(ptraj_p_norm[i][1] - etraj_p_norm[i][1]),
                                         etraj_z_sp[i])
                else:
                    etraj_p_norm[i,:] = (0,0,0)
                    ptraj_p_norm[i,:] = (0,0,0)
                    ptraj_sp[i,:]     = (0,0,0)

            # Calc z relative to ecal face
            """
            etraj_ref = np.zeros((len(etraj_p_norm), 2, 3), dtype='float32')  # Note the 2:  Only storing start and pvec_norm
            ptraj_ref = np.zeros((len(etraj_p_norm), 2, 3), dtype='float32')
            # Format is [event#] x [start of traj/p_norm] x [etraj_xyz]
            for i in range(len(etraj_p_norm)):
                etraj_ref[i][0][0] = etraj_x_sp[i]
                etraj_ref[i][0][1] = etraj_y_sp[i]
                etraj_ref[i][0][2] = etraj_z_sp[i]
                etraj_ref[i][1][0] = etraj_p_norm[i][0]
                etraj_ref[i][1][1] = etraj_p_norm[i][1]
                etraj_ref[i][1][2] = etraj_p_norm[i][2]
                ptraj_ref[i][0][0] = ptraj_sp[i][0]
                ptraj_ref[i][0][1] = ptraj_sp[i][1]
                ptraj_ref[i][0][2] = ptraj_sp[i][2]
                ptraj_ref[i][1][0] = ptraj_p_norm[i][0]
                ptraj_ref[i][1][1] = ptraj_p_norm[i][1]
                ptraj_ref[i][1][2] = ptraj_p_norm[i][2]
            table['etraj_ref'] = etraj_ref
            table['ptraj_ref'] = ptraj_ref
            """
            table['etraj_sp'] = etraj_sp
            table['ptraj_sp'] = ptraj_sp
            table['enorm_sp'] = etraj_p_norm
            table['pnorm_sp'] = ptraj_p_norm
            
            print("Finished loading coord ref")
            #print("Usage after coord ref: {}".format(psutil.virtual_memory().percent))


        def _load_recoil_pt(t, table):
            if len(obs_branches):
                # Note:  0.177 value may be wrong...but should be first SP after target.
                el = (t['TargetScoringPlaneHits_v12.pdgID_'].array() == 11) * \
                     (t['TargetScoringPlaneHits_v12.z_'].array() > 0.176) * \
                     (t['TargetScoringPlaneHits_v12.z_'].array() < 0.178) * \
                     (t['TargetScoringPlaneHits_v12.pz_'].array() > 0)
                
                tmp = np.sqrt(t['TargetScoringPlaneHits_v12.px_'].array()[el] ** 2 + t['TargetScoringPlaneHits_v12.py_'].array()[el] ** 2)
                tmp = awkward.pad_none(tmp, 1, clip=True)
                otmp = awkward.fill_none(tmp, -999)
                table['TargetSPRecoilE_pt'] = awkward.flatten(tmp)
        
        def _read_file(t, table):
            #print("    Usage before read file: {}".format(psutil.virtual_memory().percent))
            # load data from one file
            start, stop = [int(x * len(table[self._branches[0]])) for x in load_range]
            #print("start, stop: ", (start, stop))
            for k in table:
                table[k] = table[k][start:stop]
            n_inclusive = len(table[self._branches[0]])  # before preselection
            
            if apply_preselection:
                pos_pass_presel = awkward.sum(table[self._energy_branch] > 0, axis=1) < MAX_NUM_ECAL_HITS
                # NEW:
                pos_pass_presel = (table['EcalVeto_v12.summedTightIso_'] < MAX_ISO_ENERGY) * pos_pass_presel
                for k in table:
                    table[k] = table[k]#[pos_pass_presel]
            #n_selected = len(table[self._branches[0]])  # after preselection
            #print("EVENTS BEFORE PRESELECTION (in _read_file):  {}".format(n_inclusive))
            #print("EVENTS AFTER PRESELECTION: ", n_selected)

            #if n_selected == 0:   #Ignore this file
            #    print("ERROR:  ParticleNet can't handle files with no events passing selection!")

            ### Creating our recoilX, recoilY, recoilPx, recoilPy, recoilPz arrays ###          
            def _pad_array(arr):
                arr = awkward.pad_none(arr, 1, clip=True)
                arr = awkward.fill_none(arr, 0)
                return np.array(awkward.flatten(arr))  #NEW:  Include np conversion to allow stacking

            el = (t['EcalScoringPlaneHits_v12.pdgID_'].array() == 11) * \
                 (t['EcalScoringPlaneHits_v12.z_'].array() > 240.0) * \
                 (t['EcalScoringPlaneHits_v12.z_'].array() < 241.001) * \
                 (t['EcalScoringPlaneHits_v12.trackID_'].array() == 1) * \
                 self.el_ 
    
            recoilX = _pad_array(t['EcalScoringPlaneHits_v12.x_'].array()[el])[start:stop]#[pos_pass_presel]
            recoilY = _pad_array(t['EcalScoringPlaneHits_v12.y_'].array()[el])[start:stop]#[pos_pass_presel]
            recoilPx = _pad_array(t['EcalScoringPlaneHits_v12.px_'].array()[el])[start:stop]#[pos_pass_presel]
            recoilPy = _pad_array(t['EcalScoringPlaneHits_v12.py_'].array()[el])[start:stop]#[pos_pass_presel]
            recoilPz = _pad_array(t['EcalScoringPlaneHits_v12.pz_'].array()[el])[start:stop]#[pos_pass_presel]
            
            print("Usage before fiducial loop: {}".format(psutil.virtual_memory().percent))

            ### LOOPING THROUGH EACH EVENT TO MAKE A BOOLEAN ARRAY THAT SELECTS ONLY NON-FIDUCIAL ELECTRONS ###
            N = len(recoilX)
               
            simEvents = np.zeros(N, dtype=bool)
	    
            cells = np.array(list(self._cellMap.values()))
            
            for event in range(N):
                                          
                fiducial = False
                
                fXY = projection(recoilX[event], recoilY[event], scoringPlaneZ, recoilPx[event], recoilPy[event], recoilPz[event], ecalFaceZ)
                
                # If there is a hit of an event that satisfies the constraint, perform the fiducial calculation 
                if not recoilX[event] == -9999 and not recoilY[event] == -9999 and not recoilPx[event] == -9999 and not recoilPy[event] == -9999:
                    for cell in range(len(cells)):
                        celldis = dist(cells[cell], fXY)             
                        if celldis <= cell_radius:
                            fiducial = True
                            break
                
                # For events where no hits satisfy the constraints, we mark that as non-fiducial
                if recoilX[event] == 0 and recoilY[event] == 0 and recoilPx[event] == 0 and recoilPy[event] == 0 and recoilPz[event] == 0: 
                    fiducial = False
                   
                if fiducial == False:
                    simEvents[event] = 1

            #print("The number of events before the fiducial cut: " + str(len(table[self._energy_branch])))
            
            
            ### APPLYING simEvents TO THE SELECTION ###  
            for k in table:
                table[k] = table[k][simEvents]
             
            #print("The number of events after fiducial cut: " + str(len(table[self._energy_branch])))
            
            print("Usage after fiducial loop, before array creation: {}".format(psutil.virtual_memory().percent))

            eid = table[self._id_branch]
            energy = table[self._energy_branch]
            pos = (energy > 0)
            eid = eid[pos]  # Gets rid of all (AND ONLY) hits with 0 energy
            energy = energy[pos]
            x, y, z, layer_id = self._parse_cid(eid)  # layer_id > 0, so can use layer_id-1 to index e/ptraj_ref
            
            print("Usage after array creation, before trigger cut: {}".format(psutil.virtual_memory().percent))
             
            ### APPLY THE TRIGGER CUT ###
            
            #print("The number of non-fiducial events before the trigger cut: "  + str(len(energy))) 
            

            t_cut = np.zeros(len(eid), dtype = bool) # Boolean array for trigger cut: ex -> [ 1, 0, 1, 1,  0 ... ]

            for event in range(len(eid)): # Loop through each event in eid: ex -> [[EVENT 1 HITS], [EVENT 2 HITS], ...]
                en = 0.0 # Initial energy starts at 0 MeV
                
                for hit in range(len(eid[event])): # Loop through each hit of each event in eid
                     if layer_id[event][hit] < 20.0: # Check if the layer for the nth hit is less than 20
                         en += energy[event][hit] # Add that hit's corresponding energy from the energy-array to the total energy "en"
                if en < 1500.0: # If the energy is less than 1500.0 MeV after looping through the first 20 layers, mark as True (we keep this event)
                    t_cut[event] = 1    
                        
            # We apply the trigger cut to the eid, energy, x, y, z, layer_id arrays 
            eid = eid[t_cut]                 
            energy = energy[t_cut] 
            x = x[t_cut]
            y = y[t_cut]
            z = z[t_cut]
            layer_id = layer_id[t_cut]
            
            print("Usage after trigger cut, before creating x_e, y_e ...: {}".format(psutil.virtual_memory().percent))
                   
            #print("The number of non-fiducial events after the trigger cut: "  + str(len(energy)))            

            #print("The total number of events before the trigger cut: "  + str(len(energy2)))
            '''
            t_cut2 = np.zeros(len(eid2), dtype = bool) # Boolean array for trigger cut: ex -> [ 1, 0, 1, 1,  0 ... ]

            for event2 in range(len(eid2)): # Loop through each event in eid: ex -> [[EVENT 1 HITS], [EVENT 2 HITS], ...]
                en2 = 0.0 # Initial energy starts at 0 MeV

                for hit2 in range(len(eid2[event2])): # Loop through each hit of each event in eid
                     if layer_id2[event2][hit2] < 20.0: # Check if the layer for the nth hit is less than 20
                         en2 += energy2[event2][hit2] # Add that hit's corresponding energy from the energy-array to the total energy "en"
                if en2 < 1500.0: # If the energy is less than 1500.0 MeV after looping through the first 20 layers, mark as True (we keep this event)
                    t_cut2[event2] = 1

            # We apply the trigger cut to the eid, energy, x, y, z, layer_id arrays 
            eid2 = eid2[t_cut2]
            energy2 = energy2[t_cut2]
            x2 = x2[t_cut2]
            y2 = y2[t_cut2]
            z2 = z2[t_cut2]
            layer_id2 = layer_id2[t_cut2]
            '''
            #print("The total number of events after the trigger cut: "  + str(len(energy2)))
            

            n_selected = len(energy)

            # Now, work with table['etraj_ref'] and table['ptraj_ref'].
            # Create lists:  x/y/z_e, p
            # For each event, look through all hits.
            # - Determine whether hit falls inside either the e or p RoCs
            # - If so, fill corresp xyzlayer, energy, eid lists...
            x_e =           np.zeros((len(x), MAX_NUM_ECAL_HITS), dtype='float32')  # In theory, can lower size of 2nd dimension...
            y_e =           np.zeros((len(x), MAX_NUM_ECAL_HITS), dtype='float32')
            z_e =           np.zeros((len(x), MAX_NUM_ECAL_HITS), dtype='float32')
            log_energy_e =  np.zeros((len(x), MAX_NUM_ECAL_HITS), dtype='float32')
            layer_id_e =    np.zeros((len(x), MAX_NUM_ECAL_HITS), dtype='float32')
        
            print("    Usage after creating x_e, y_e ...: {}".format(psutil.virtual_memory().percent))
            
            for i in range(len(x)):  # For every event...
                etraj_sp = table['etraj_sp'][i]  #table['etraj_ref'][i][0]  # e- location at scoring plane (approximate)
                enorm_sp = table['enorm_sp'][i]  #table['etraj_ref'][i][1]  # normalized (dz=1) momentum = direction of trajectory
                ptraj_sp = table['ptraj_sp'][i]  #table['ptraj_ref'][i][0]
                pnorm_sp = table['pnorm_sp'][i]  #table['ptraj_ref'][i][1]
                for j in range(min(len(x[i]), MAX_NUM_ECAL_HITS)):  #range(MAX_NUM_ECAL_HITS):  # For every hit...
                    layer_index = int(layer_id[i][j])
                    # Calculate xy coord of point on projected trajectory in same layer
                    delta_z = self._layerZs[layer_index] - etraj_sp[2]
                    etraj_point = (etraj_sp[0] + enorm_sp[0]*delta_z, etraj_sp[1] + enorm_sp[1]*delta_z)
                    ptraj_point = (ptraj_sp[0] + pnorm_sp[0]*delta_z, ptraj_sp[1] + pnorm_sp[1]*delta_z)
                    # Additionally, calculate recoil angle (angle of pnorm_sp):
                    recoilangle = enorm_sp[2] / np.sqrt(enorm_sp[0]**2 + enorm_sp[1]**2 + enorm_sp[2]**2)
                    recoil_p = np.sqrt(enorm_sp[0]**2 + enorm_sp[1]**2 + enorm_sp[2]**2)
                    ir = -1
                    #if recoilangle==-1 or recoil_p==-1:  ir = 1  # Not used for now
                    if recoilangle<10 and recoil_p<500:
                        ir = 1
                    elif recoilangle<10 and recoil_p >= 500:
                        ir = 2
                    elif recoilangle<=20:
                        ir = 3
                    else:
                        ir = 4
                    # Determine what regions the hit falls into:
                    insideElectronRadius = np.sqrt((etraj_point[0] - x[i][j])**2 + \
                            (etraj_point[1] - y[i][j])**2) < 1.0 * radius_68[ir][layer_index]
                    insidePhotonRadius   = np.sqrt((ptraj_point[0] - x[i][j])**2 + \
                            (ptraj_point[1] - y[i][j])**2) < 1.0 * radius_68[ir][layer_index]
                    # NEW:  If an SP electron hit is missing, place all hits in the event into the "other" region
                    # 3-region:
                    if enorm_sp[0] == 0 and enorm_sp[1] == 0:
                        insideElectronRadius = False
                        insidePhotonRadius   = False
                    
                    insideElectronRadius = True
                    if insideElectronRadius:
                        x_e[i][j] = x[i][j] - etraj_point[0]  # Store coordinates relative to the xy distance from the trajectory
                        y_e[i][j] = y[i][j] - etraj_point[1]
                        z_e[i][j] = z[i][j] - self._layerZs[0]  # Defined relative to the ecal face
                        log_energy_e[i][j] = np.log(energy[i][j]) if energy[i][j] > 0 else 0
                        layer_id_e[i][j] = layer_id[i][j]
            #print("    Usage after region determination: {}".format(psutil.virtual_memory().percent))        

            var_dict = {'log_energy_e':log_energy_e,
                        'x_e':x_e, 'y_e':y_e, 'z_e':z_e, 'layer_id_e':layer_id_e,
                       }

            obs_dict = {k: table[k] for k in obs_branches + ecal_veto_branches}

            return (n_inclusive, n_selected), var_dict, obs_dict

        def _load_dataset(filelist, name):
            # load data from all files in the siglist or bkglist
            n_sum = 0
            for extra_label in filelist:
                filepath, max_event = filelist[extra_label]
                if len(glob.glob(filepath)) == 0:
                    print('No matches for filepath %s: %s, skipping...' % (extra_label, filepath))
                    return
                if ignore_evt_limits:
                    max_event = -1
                n_total_inclusive = 0
                n_total_selected = 0
                var_dict = {}
                obs_dict = {k:[] for k in obs_branches + ecal_veto_branches}
                # NEW:  Dictionary storing particle data for e/p trajectory
                # Want position, momentum of e- hit; calc photon info from it
                spHit_dict = {}
                print('Start loading dataset %s (%s)' % (filepath, name))

                with tqdm.tqdm(glob.glob(filepath)) as tq:
                    for fp in tq:
                        #print("    Usage before file load: {}".format(psutil.virtual_memory().percent))
                        t = uproot.open(fp)['LDMX_Events']
                        if len(t.keys()) == 0:
#                             print('... ignoring empty file %s' % fp)
                            continue
                        load_branches = [k for k in self._branches + obs_branches if '.' in k and k[-1] == '_']
                        table_temp = t.arrays(expressions=load_branches, interpretation_executor=executor)  #, library="ak")
                        table = {}
                        for k in load_branches:
                            table[k] = table_temp[k]


                        # Now go through and load Ecal branches separately.
                        # New branch for cut:
                        EcalVeto = t["EcalVeto_v12"]
                        #table["EcalVeto_v12.summedTightIso_"] = EcalVeto["summedTightIso_"].array(interpretation_executor=executor)
                        # All other ecal branches:
                        if ecal_veto_branches:  # Was veto_branches; also commented the summedTightIso line
                            for branch in ecal_veto_branches:
                                #table["EcalVeto_v12."+branch] = EcalVeto[branch].array(interpretation_executor=executor)
                                table[branch] = EcalVeto[branch.split('.')[1]].array(interpretation_executor=executor)

                        _load_coord_ref(t, table)
                        _load_recoil_pt(t, table)

                        (n_inc, n_sel), v_d, o_d = _read_file(t, table)

                        n_total_inclusive += n_inc
                        n_total_selected += n_sel
                        print("N_SELECTED:  ", n_sel)
                        print("TOTAL SELECTED:  ", n_total_selected)

                        for k in v_d:
                            if k in var_dict: # If the key already exists, we add to that key's array
                                var_dict[k].append(v_d[k])
                            else:
                                var_dict[k] = [v_d[k]] # If the key doesn't exist, we put the array in
                        for k in obs_dict:
                            obs_dict[k].append(o_d[k])
                        if max_event > 0 and n_total_selected >= max_event:
                            break

                        #print("    Usage after loaded file: {}".format(psutil.virtual_memory().percent))
                        gc.collect()  # May reduce RAM usage

                # calc preselection eff before dropping events more than `max_event`
                self.presel_eff[extra_label] = float(n_total_selected) / n_total_inclusive
                # now we concat the arrays and remove the extra events if needed
                n_total_loaded = None
                upper = None
                if max_event > 0 and max_event < n_total_selected:
                    upper = max_event - n_total_selected
               
              #  print("var_dict: " + str(var_dict))
                
                for k in var_dict:
                    var_dict[k] = _concat(var_dict[k])[:upper]

              #      if k == 'log_energy_e':
              #          print("The length of var_dict is: " + str(len(var_dict[k])))

                    if n_total_loaded is None:
                        n_total_loaded = len(var_dict[k])
               #         print("n_total_loaded: " + str(n_total_loaded))

               #     else:
               #         assert(n_total_loaded == len(var_dict[k]))
                for k in obs_dict:
                    obs_dict[k] = _concat(obs_dict[k])[:upper]
               #     assert(n_total_loaded == len(obs_dict[k]))
                print('Total %d events, selected %d events, finally loaded %d events.' % (n_total_inclusive, n_total_selected, n_total_loaded))

                self.extra_labels.append(extra_label * np.ones(n_total_loaded, dtype='int32'))
                for k in var_dict:
                    if k in self.var_data:
                        self.var_data[k].append(var_dict[k])
                    else:
                        self.var_data[k] = [var_dict[k]]
                for k in obs_branches + ecal_veto_branches:
                    self.obs_data[k].append(obs_dict[k])
                n_sum += n_total_loaded
                
                gc.collect()
                #print("Usage after load: {}".format(psutil.virtual_memory().percent))
                print("RETURNING", n_sum)
            return n_sum

        nsig = _load_dataset(siglist, 'sig')
        nbkg = _load_dataset(bkglist, 'bkg')
        print("Preparing to train on {} background events, {} (total) signal events".format(nbkg, nsig)) 

       # label for training
        self.label = np.zeros(nsig + nbkg, dtype='float32')
        self.label[:nsig] = 1

        self.extra_labels = np.concatenate(self.extra_labels)
        for k in self.var_data:
            self.var_data[k] = _concat(self.var_data[k])
        for k in obs_branches + ecal_veto_branches:
            self.obs_data[k] = _concat(self.obs_data[k])

        # training features
        # Multiple regions:
        """
        coords_e = np.stack((self.var_data['x_e'], self.var_data['y_e'], self.var_data['z_e']), axis=1)
        coords_p = np.stack((self.var_data['x_p'], self.var_data['y_p'], self.var_data['z_p']), axis=1)
        coords_o = np.stack((self.var_data['x_o'], self.var_data['y_o'], self.var_data['z_o']), axis=1)
        self.coordinates = np.stack((coords_e, coords_p, coords_o))
        del coords_e
        del coords_p
        del coords_o
        features_e = np.stack((self.var_data['x_e'], self.var_data['y_e'], self.var_data['z_e'], self.var_data['layer_id_e'], self.var_data['log_energy_e']), axis=1)
        features_p = np.stack((self.var_data['x_p'], self.var_data['y_p'], self.var_data['z_p'], self.var_data['layer_id_p'], self.var_data['log_energy_p']), axis=1)
        features_o = np.stack((self.var_data['x_o'], self.var_data['y_o'], self.var_data['z_o'], self.var_data['layer_id_o'], self.var_data['log_energy_o']), axis=1)
        self.features    = np.stack((features_e, features_p, features_o))
        del features_e
        del features_p
        del features_o
        """
        # 1 region:
        self.coordinates = np.stack((self.var_data['x_e'], self.var_data['y_e'], self.var_data['z_e']), axis=1)
        self.features    = np.stack((self.var_data['x_e'], self.var_data['y_e'], self.var_data['z_e'],
                                     self.var_data['layer_id_e'], self.var_data['log_energy_e']), axis=1)
        #assert(len(self.coordinates) == len(self.label))
        #assert(len(self.features) == len(self.label))

        # NEW:  Free up old variables after the coords and features have been assigned
        #for key, item in self.var_data.items():
        #    del item
        #for key, item in self.obs_data.items():
        #    del item
        gc.collect()
        #print("Usage after coord+feature creation: {}".format(psutil.virtual_memory().percent))


    def _load_cellMap(self, version='v12'):
        self._cellMap = {}
        for i, x, y in np.loadtxt('data/%s/cellmodule.txt' % version):
            self._cellMap[i] = (x, y)
        self._layerZs = np.loadtxt('data/%s/layer.txt' % version)
        print("Loaded detector info")

    def _parse_cid(self, cid):  # Retooled for v12
        # For id details, see (?):  DetDescr/src/EcalID.cxx
        # Flatten arrays to 1D numpy arrays so zip, map will work
        cell   = (awkward.to_numpy(awkward.flatten(cid)) >> 0)  & 0xFFF
        module = (awkward.to_numpy(awkward.flatten(cid)) >> 12) & 0x1F
        layer  = (awkward.to_numpy(awkward.flatten(cid)) >> 17) & 0x3F
        
        mcid = 10 * cell + module
        x, y = zip(*map(self._cellMap.__getitem__, mcid))
        z = list(map(self._layerZs.__getitem__, layer))

        def unflatten_array(x, base_array):
            # x = 1D flattened np array, base_array has the desired shape
            return awkward.Array(awkward.layout.ListOffsetArray32(
                                    awkward.layout.Index32(base_array.layout.offsets),   # NOTE, may need to change to offsets32
                                    awkward.layout.NumpyArray(np.array(x, dtype='float32'))
                                    )
                                )
        x        = unflatten_array(x, cid)
        y        = unflatten_array(y, cid)
        z        = unflatten_array(z, cid)
        layer_id = unflatten_array(layer, cid)

        return x, y, z, layer_id

    @property
    def num_features(self):
        return self.features.shape[1]
        #return self.features.shape[2]  # Modified

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):  # NOTE:  This now returns e/p data.  May need modification.
        pts = self.coordinates[i]
        fts = self.features[i]
        y = self.label[i]
        return pts, fts, y


class _SimpleCustomBatch:

    def __init__(self, data, min_nodes=None):
        pts, fts, labels = list(zip(*data))
        self.coordinates = torch.tensor(pts)
        self.features = torch.tensor(fts)
        self.label = torch.tensor(labels)

    def pin_memory(self):
        self.coordinates = self.coordinates.pin_memory()
        self.features = self.features.pin_memory()
        self.label = self.label.pin_memory()
        return self


def collate_wrapper(batch):
    return _SimpleCustomBatch(batch)
