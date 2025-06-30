import argparse
import os
from pprint import pprint
import shutil
import pathlib
import subprocess
import numpy as np


def get_ridft_path():
    turbodir = pathlib.Path(os.environ["TURBODIR"])
    return turbodir/"bin"/"em64t-unknown-linux-gnu_smp"/"ridft_omp"

def run_t2x():
    turbodir = pathlib.Path(os.environ["TURBODIR"])
    t2x = turbodir/"scripts"/"t2x"
    os.system(f"{t2x} coord > coord.xyz")

def read_exspectrum():
    data = open("exspectrum").readlines()
    return [[float(ls.split()[3]),float(ls.split()[7])] for i,ls in enumerate(data) if len(ls.split()) == 8 and i > 1 ]

def find_ct_excitations(dimer_only):
    if os.path.exists("escf.out"):
        fname = "escf.out"
    elif os.path.exists("egrad.out"):
        fname = "egrad.out"
    else:
        raise Exception("No escf or egrad file found in directory")

    excitations = []
    reading_orb = False
    with open(fname, 'r') as inf:
        for i,line in enumerate(inf):
            ls = line.split()
            if reading_orb and len(ls) != 7:
                reading_orb = False
            if reading_orb:
                excitations[-1].append([int(ls[0]),int(ls[3]),float(ls[6])])

            if len(ls) > 2 and ls[0] == "occ." and ls[1] == "orbital":
                excitations.append([])
                reading_orb = True

    all_orbs = list(set([orb for exc in excitations for contr in exc for orb in [contr[0], contr[1]]]))
    orb_atoms = run_pop_for_orbs(all_orbs)
    ct_char, exc_locs = est_ct_character(excitations, orb_atoms, dimer_only)
    spectrum = read_exspectrum()
    
    ct_excs = []
    non_ct_excs = []
    for ct,(e,osc), ex_loc in zip(ct_char,spectrum, exc_locs):
        if ct > 0.75:
            ct_excs.append([e,osc, int(ex_loc)])
        else:
            non_ct_excs.append([e,osc, int(ex_loc)])
    if len(ct_excs) > 0:
        np.savetxt("ct_excitations.dat", ct_excs)
    if len(non_ct_excs) > 0:
        np.savetxt("non_ct_excitations.dat", non_ct_excs)

def est_ct_character(exc_orbs, orb_atoms, dimer_only=False):
    if dimer_only:
        from split_xyz_dimer import read_xyz,find_monomers
        run_t2x()
        atoms = read_xyz("coord.xyz")
        monomer1_ids, monomer2_ids = find_monomers(atoms)
        if len(monomer2_ids) == 0:
            raise Exception("user requested dimer ct state but could not identify dimer fragments")
    
    ct_character = []
    exc_locs = []
    for excs in exc_orbs:
        new_w = 0.0
        kept_w = 0.0
        total_ct_amount = 0.0
        total_weight = 0.0
        exc_loc = {"11": 0.0, "22": 0.0, "12": 0.0, "21": 0.0}
        for exc in excs:
            iorb, forb, w = exc
            atom_ids_i = np.array([ato[0] for ato in orb_atoms[iorb]])
            atom_ids_f = np.array([ato[0] for ato in orb_atoms[forb]])
            atom_w_f = {ato[0]: ato[2] for ato in orb_atoms[forb]}
            atom_w_i = {ato[0]: ato[2] for ato in orb_atoms[iorb]}
            if dimer_only:
                #we are looking for dimer ct excitations:
                hole_on_m1  = sum(atom_w_i.get(idx,0.0) for idx in atom_ids_i if idx in monomer1_ids )
                hole_on_m2  = sum(atom_w_i.get(idx,0.0) for idx in atom_ids_i if idx in monomer2_ids )
                electron_on_m1 = sum(atom_w_f.get(idx,0.0) for idx in atom_ids_f if idx in monomer1_ids )
                electron_on_m2 = sum(atom_w_f.get(idx,0.0) for idx in atom_ids_f if idx in monomer2_ids )
                
                #if init_1 > init_2  and final_2 > final_1:
                #    new_w += (final_2 / (init_2 +final_2)) * w/100.0
                #    kept_w += (init_2 / (init_2 +final_2)) * w/100.0
                #elif init_1 < init_2  and final_2 < final_1:
                #    new_w += (final_1 / (init_1 +final_1)) * w/100.0
                #    kept_w += (init_1 / (init_1 +final_1)) * w/100.0
                #else:
                #    new_w += 0.0 * w
                #    kept_w += 1.0 * w
                ct_1_to_2 = min(hole_on_m1, electron_on_m2)
                ct_2_to_1 = min(hole_on_m2, electron_on_m1)
                exc_loc["11"] += min(hole_on_m1, electron_on_m1) * w
                exc_loc["22"] += min(hole_on_m2, electron_on_m2) * w
                exc_loc["12"] += ct_1_to_2 * w
                exc_loc["21"] += ct_2_to_1 * w

                # The total CT for this single orbital transition is the sum of both directions
                transition_ct = ct_1_to_2 + ct_2_to_1

                # Add the weighted CT amount to the total for the excitation
                total_ct_amount += w * transition_ct
                total_weight += w

            else:
                new_atoms = np.setdiff1d(atom_ids_f, atom_ids_i)
                kept_atoms = np.intersect1d(atom_ids_f, atom_ids_i)
                new_w += sum(atom_w_f.get(idx,0.0) for idx in new_atoms) * w/100.0
                kept_w += sum(atom_w_f.get(idx,0.0) for idx in kept_atoms) * w/100.0
        if dimer_only:
            if total_weight > 1e-6:
                ct_character.append(total_ct_amount / total_weight)
            else:
                ct_character.append(0.0)
            exc_locs.append(max(exc_loc,key=exc_loc.get))
        else:
            ct_character.append(new_w/(new_w+kept_w))
            exc_locs.append(0.0)

    return ct_character, exc_locs


def run_pop_for_orbs(orb_ids):
    shutil.copyfile("control", "control_backup")
    cf = open("control").readlines()
    with open("control", 'w') as outf:
        for line in cf:
            if "$end" in line:
                outf.write(f"$pop mo {','.join([str(orb) for orb in orb_ids])}\n")
            outf.write(line)
    ridft_path = get_ridft_path()
    os.system(f"{ridft_path} -proper > proper_mos.out")
    shutil.move("control_backup","control")
    orb_atoms = parse_orb_atoms("proper_mos.out")
    return orb_atoms

def parse_orb_atoms(proper_file):
    with open(proper_file) as inf:
        start = False
        read_orb = False
        orbs =  {}
        for line in inf:
            ls = line.split()
            if "MULLIKEN BRUTTO POPULATIONS FOR SELECTED MOS" in line:
                start = True
            if read_orb and len(ls) ==0:
                read_orb = False
            if read_orb and len(ls) >= 2 and ls[0] != "atom":
                atom_id = int("".join([char for char in ls[0] if char.isdigit()]))
                atom_el = "".join([char for char in ls[0] if not char.isdigit()])
                orbs[orb_id].append([atom_id, atom_el, float(ls[1])])
            if start and len(ls) == 3 and "energy" in ls[1]:
                orb_id = int("".join([char for char in ls[0] if char.isdigit()]))
                read_orb = True
                orbs[orb_id] = []
    return orbs





def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Parsers for various TM outputs ')
    parser.add_argument('--find_ct_exc', action='store_true', help='creates seperate file for excitation with CT character and without CT character')
    parser.add_argument('--dimer_ct', action='store_true', help='consider only dimer CT as CT')
    args = parser.parse_args()

    if args.find_ct_exc:
        find_ct_excitations(args.dimer_ct)


if __name__ == "__main__":
    main()

