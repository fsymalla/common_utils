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

def read_exspectrum():
    data = open("exspectrum").readlines()
    return [[float(ls.split()[3]),float(ls.split()[7])] for i,ls in enumerate(data) if len(ls.split()) == 8 and i > 1 ]

def find_ct_excitations():
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
    ct_char = est_ct_character(excitations, orb_atoms)
    spectrum = read_exspectrum()
    
    ct_excs = []
    non_ct_excs = []
    for ct,(e,osc) in zip(ct_char,spectrum):
        if ct > 0.75:
            ct_excs.append([e,osc])
        else:
            non_ct_excs.append([e,osc])
    if len(ct_excs) > 0:
        np.savetxt("ct_excitations.dat", ct_excs)
    if len(non_ct_excs) > 0:
        np.savetxt("non_ct_excitations.dat", non_ct_excs)

def est_ct_character(exc_orbs, orb_atoms):
    ct_character = []
    for excs in exc_orbs:
        new_w = 0.0
        kept_w = 0.0
        for exc in excs:
            iorb, forb, w = exc
            atom_ids_i = np.array([ato[0] for ato in orb_atoms[iorb]])
            atom_ids_f = np.array([ato[0] for ato in orb_atoms[forb]])
            atom_w_f = {ato[0]: ato[2] for ato in orb_atoms[forb]}
            new_atoms = np.setdiff1d(atom_ids_f, atom_ids_i)
            kept_atoms = np.intersect1d(atom_ids_f, atom_ids_i)
            new_w += sum(atom_w_f.get(idx,0.0) for idx in new_atoms) * w/100.0
            kept_w += sum(atom_w_f.get(idx,0.0) for idx in kept_atoms) * w/100.0
        ct_character.append(new_w/(new_w+kept_w))

    return ct_character


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
            if read_orb and len(ls) < 5:
                read_orb = False
            if read_orb and len(ls) >= 5 and ls[0] != "atom":
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
    args = parser.parse_args()

    if args.find_ct_exc:
        find_ct_excitations()


if __name__ == "__main__":
    main()

