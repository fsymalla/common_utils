import math
import argparse
from collections import deque

# A dictionary of approximate covalent radii (in Angstroms) for common elements
# This can be used for a more adaptive cutoff, but a fixed cutoff is simpler for a start.
# For this script, we'll start with a fixed cutoff.
# COVALENT_RADII = {
#     'H': 0.37, 'C': 0.77, 'N': 0.75, 'O': 0.73, 'F': 0.71,
#     'P': 1.10, 'S': 1.03, 'Cl': 0.99,
#     # Add more as needed
# }

def read_xyz(filename):
    """
    Reads an XYZ file and returns a list of atoms.
    Each atom is a dictionary: {'id': int, 'element': str, 'coords': [x, y, z]}
    """
    atoms = []
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            num_atoms = int(lines[0].strip())
            # print(f"Expecting {num_atoms} atoms from file.")
            if len(lines) < num_atoms + 2:
                raise ValueError("XYZ file is shorter than expected from atom count.")

            for i in range(num_atoms):
                line_num = i + 2
                parts = lines[line_num].strip().split()
                if len(parts) < 4:
                    print(f"Warning: Line {line_num+1} in {filename} is malformed: {lines[line_num].strip()}")
                    continue
                try:
                    element = parts[0]
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    # Atom IDs are 1-based, corresponding to their order in the file
                    atoms.append({'id': i + 1, 'element': element, 'coords': [x, y, z]})
                except ValueError:
                    print(f"Warning: Could not parse coordinates on line {line_num+1}: {lines[line_num].strip()}")
                    continue
            
            if len(atoms) != num_atoms:
                print(f"Warning: Read {len(atoms)} atoms, but expected {num_atoms} from header.")

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None
    except ValueError as e:
        print(f"Error parsing XYZ file: {e}")
        return None
    except IndexError:
        print(f"Error: File '{filename}' seems to have an unexpected format or is incomplete.")
        return None
    return atoms

def calculate_distance_sq(atom1_coords, atom2_coords):
    """Calculates the squared Euclidean distance between two sets of coordinates."""
    return sum([(c1 - c2)**2 for c1, c2 in zip(atom1_coords, atom2_coords)])

def find_monomers(atoms, connectivity_cutoff_A=2.5):
    """
    Finds monomers based on atom connectivity.

    Args:
        atoms (list): List of atom dictionaries from read_xyz.
        connectivity_cutoff_A (float): Distance in Angstroms to consider atoms connected.

    Returns:
        tuple: (monomer1_ids, monomer2_ids) or (None, None) if not a dimer.
               Returns (all_atom_ids, []) if only one component is found.
    """
    if not atoms:
        return None, None

    num_atoms = len(atoms)
    adj = {atom['id']: [] for atom in atoms} # Adjacency list using atom IDs

    # Use atom['id'] which is 1-based
    atom_map = {atom['id']: atom for atom in atoms} 

    cutoff_sq = connectivity_cutoff_A**2

    # Build adjacency list
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            atom_i = atoms[i]
            atom_j = atoms[j]
            dist_sq = calculate_distance_sq(atom_i['coords'], atom_j['coords'])
            if dist_sq < cutoff_sq:
                adj[atom_i['id']].append(atom_j['id'])
                adj[atom_j['id']].append(atom_i['id'])

    # Find connected components using BFS
    visited = set()
    components = []
    for atom_id in atom_map.keys(): # Iterate using the 1-based IDs
        if atom_id not in visited:
            current_component = []
            q = deque([atom_id])
            visited.add(atom_id)
            while q:
                u = q.popleft()
                current_component.append(u)
                for v_neighbor_id in adj[u]:
                    if v_neighbor_id not in visited:
                        visited.add(v_neighbor_id)
                        q.append(v_neighbor_id)
            if current_component:
                components.append(sorted(current_component))

    # Sort components by size (largest first)
    components.sort(key=len, reverse=True)

    if not components:
        print("No components found (empty structure or issue).")
        return None, None
    
    print(f"Found {len(components)} component(s):")
    for i, comp in enumerate(components):
        print(f"  Component {i+1}: {len(comp)} atoms, IDs: {comp[:5]}..." if len(comp) > 5 else f"  Component {i+1}: {len(comp)} atoms, IDs: {comp}")


    if len(components) == 1:
        print("Only one component found. Assuming it's a single molecule.")
        return components[0], [] # Return all atoms as monomer1, monomer2 as empty
    elif len(components) >= 2:
        print("Multiple components found. Assuming the two largest are the monomers of a dimer.")
        return components[0], components[1]
    else: # Should not happen if len(components) == 0 handled
        print("Could not identify two distinct monomers.")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Finds atom IDs of two monomers in an XYZ file representing a dimer.")
    parser.add_argument("xyz_file", help="Path to the XYZ input file.")
    parser.add_argument(
        "-c", "--cutoff", type=float, default=2.1,
        help="Connectivity cutoff distance in Angstroms (default: 2.5). "
             "Atoms closer than this are considered part of the same monomer."
    )
    args = parser.parse_args()

    print(f"Reading XYZ file: {args.xyz_file}")
    atoms = read_xyz(args.xyz_file)

    if atoms:
        print(f"Successfully read {len(atoms)} atoms.")
        print(f"Using connectivity cutoff: {args.cutoff} Å")
        
        monomer1_ids, monomer2_ids = find_monomers(atoms, args.cutoff)

        if monomer1_ids and monomer2_ids:
            print("\n--- Monomer Identification ---")
            print(f"Monomer A ({len(monomer1_ids)} atoms):")
            print(f"  Atom IDs: {sorted(monomer1_ids)}")
            print(f"\nMonomer B ({len(monomer2_ids)} atoms):")
            print(f"  Atom IDs: {sorted(monomer2_ids)}")
            
            # Sanity check: total atoms
            total_identified_atoms = len(monomer1_ids) + len(monomer2_ids)
            if total_identified_atoms != len(atoms):
                 print(f"\nWarning: Total atoms in identified monomers ({total_identified_atoms}) "
                       f"does not match total atoms in file ({len(atoms)}). "
                       "This might happen if there are more than two components (e.g., solvent).")
            else:
                print("\nAll atoms assigned to the two largest monomers.")

        elif monomer1_ids and not monomer2_ids: # Single molecule case
            print("\n--- Monomer Identification ---")
            print(f"Monomer A / Single Molecule ({len(monomer1_ids)} atoms):")
            print(f"  Atom IDs: {sorted(monomer1_ids)}")
        else:
            print("\nCould not reliably identify monomers.")

if __name__ == "__main__":
    main()
