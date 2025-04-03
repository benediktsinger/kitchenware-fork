import gemmi
import numpy as np
import torch as pt
import h5py
import os
from glob import glob
from pathlib import Path

from .dtype import Structure
from .structure import split_by_chain
from .standard_encoding import std_resnames


def load_structure(filepath: str, rm_wat=False, rm_hs=True) -> Structure:
    # use gemmi to parse file
    doc = gemmi.read_structure(filepath)

    # alternative location and insertion sets
    altloc_set, resid_set = set(), set()

    # data storage
    xyz_l, element_l, name_l, resname_l, resid_l, chain_name_l = [], [], [], [], [], []

    # parse structure
    for _, model in enumerate(doc):
        for a in model.all():
            # skip hydrogens and deuterium
            if rm_hs and ((a.atom.element.name == "H") or (a.atom.element.name == "D")):
                continue

            # skip (heavy) water
            if rm_wat and ((a.residue.name == "HOH") or (a.residue.name == "DOD")):
                continue

            # altloc check (keep first encountered)
            if a.atom.has_altloc():
                key = f"{a.chain.name}_{a.residue.seqid.num}_{a.residue.name}_{a.atom.name}"
                if key in altloc_set:
                    continue
                else:
                    altloc_set.add(key)

            # insertion code (shift residue index)
            resid_set.add(
                f"{a.chain.name}_{a.residue.seqid.num}_{a.residue.seqid.icode.strip()}"
            )

            # store data
            xyz_l.append([a.atom.pos.x, a.atom.pos.y, a.atom.pos.z])
            element_l.append(a.atom.element.name)
            name_l.append(a.atom.name)
            resname_l.append(a.residue.name)
            resid_l.append(len(resid_set))
            chain_name_l.append(a.chain.name)

    # pack data
    return Structure(
        xyz=np.array(xyz_l, dtype=np.float32),
        names=np.array(name_l),
        elements=np.array(element_l),
        resnames=np.array(resname_l),
        resids=np.array(resid_l, dtype=np.int32),
        chain_names=np.array(chain_name_l),
    )


def subunit_to_pdb_str(subunit: Structure, chain_name, bfactors={}):
    # extract data
    pdb_str = ""
    for i in range(subunit.xyz.shape[0]):
        h = "ATOM" if subunit.resnames[i] in std_resnames else "HETATM"
        n = subunit.names[i]
        rn = subunit.resnames[i]
        e = subunit.elements[i]
        ri = subunit.resids[i]
        xyz = subunit.xyz[i]
        if chain_name in bfactors:
            bf = bfactors[chain_name][i]
        else:
            bf = 0.0

        # format pdb line
        pdb_str += "{:<6s}{:>5d}  {:<4s}{:>3s} {:1s}{:>4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:<2s}  \n".format(
            h, i + 1, n, rn, chain_name, ri, xyz[0], xyz[1], xyz[2], 0.0, bf, e
        )

    return pdb_str


def save_pdb(structure: Structure, filepath: str, bfactors={}):
    # split by chain
    subunits = split_by_chain(structure)

    # open file stream
    with open(filepath, "w") as fs:
        for cn in subunits:
            # get subunit
            subunit = subunits[cn]

            # convert subunit to string
            pdb_str = subunit_to_pdb_str(subunit, cn, bfactors)

            # write to file
            fs.write(pdb_str + "TER\n")

        # end of file
        fs.write("END")


def save_traj_pdb(structure: Structure, filepath):
    # split by chain
    subunits = split_by_chain(structure)
    xyz_dict = {cn: subunits[cn].xyz.copy() for cn in subunits}

    # determine number of frames
    assert len(structure.xyz.shape) == 3, "no time dimension"
    num_frames = structure.xyz.shape[0]

    # open file stream
    with open(filepath, "w") as fs:
        for k in range(num_frames):
            fs.write("MODEL    {:>4d}\n".format(k))
            for cn in subunits:
                # get subunit
                subunit = subunits[cn]

                # set coordinates to frame
                subunit.xyz = xyz_dict[cn][k]

                # convert subunit to string
                pdb_str = subunit_to_pdb_str(subunit, cn)

                # write to file
                fs.write(pdb_str + "\nTER\n")

            # end of model
            fs.write("ENDMDL\n")

        # end of file
        fs.write("END")


def fix_plinder_structure(filepath: str) -> str:
    with open(filepath, "r") as fs:
        lines = fs.readlines()

    buffer = []
    new_file = []

    for line in lines:
        if (
            line.count('"') % 2 == 1 and len(buffer) == 0
        ):  # Start buffering unclosed quotes
            buffer.append(line.replace("\n", ""))
        elif line.count('"') % 2 == 1 and len(buffer) > 0:  # Close buffered quotes
            buffer.append(line)
            fixed_line = "".join(buffer)
            buffer = []
            new_file.append(fixed_line)
        elif len(buffer) > 0:  # Continue buffering lines without quotes
            buffer.append(line.replace("\n", ""))
        else:  # Append complete lines directly
            new_file.append(line)

    # Handle leftover buffer (unmatched quotes)
    if buffer:
        print("Warning: Unmatched quotes found at end of file.")
        new_file.append("".join(buffer))

    fixed_filepath = filepath.replace(".cif", "_fixed.cif")
    with open(fixed_filepath, "w") as fs:
        fs.writelines(new_file)

    return fixed_filepath


def parse_cfw2_cif_and_compare(cif_bytes, reference_structure):
    # Convert byte array to string
    cif_str = b"".join(cif_bytes).decode()

    # Parse CIF content
    doc = gemmi.cif.read_string(cif_str)
    block = doc.sole_block()

    # Extract coordinates from CIF
    atom_site = block.find_mmcif_category("_atom_site.")
    charge_site = block.find_mmcif_category("_sb_ncbr_partial_atomic_charges")

    # Create numpy array of coordinates from CIF data
    cif_coords = []
    cif_charges = []
    cif_elements = []

    for row, entry in zip(atom_site, charge_site):
        cif_coords.append(
            [float(row["Cartn_x"]), float(row["Cartn_y"]), float(row["Cartn_z"])]
        )
        cif_elements.append(row["type_symbol"])
        cif_charges.append(entry["charge"])

    cif_coords = np.array(cif_coords)

    # asserts using the reference structure
    assert len(cif_coords) == len(
        reference_structure.xyz
    ), "Number of atoms in CIF does not match reference structure"
    assert np.allclose(
        cif_coords, reference_structure.xyz
    ), "Coordinates in CIF do not match reference structure"
    assert np.all(
        cif_elements == reference_structure.elements
    ), "Elements in CIF do not match reference structure"

    # Return the parsed coordinates for further inspection if needed
    reference_structure.charges = np.array(cif_charges, dtype=np.float32)
    return reference_structure


def parse_esp_cif_and_compare(cif_bytes, structure):
    mol2_lines = b"".join(cif_bytes).decode().split("\n")

    # Find the atom section
    atom_section_start = mol2_lines.index("@<TRIPOS>ATOM")
    atom_section_end = mol2_lines.index("@<TRIPOS>BOND")

    # Extract atom lines
    atom_lines = mol2_lines[atom_section_start + 1 : atom_section_end]

    # Create a boolean mask array
    mask = np.zeros(len(structure.charges), dtype=bool)
    new_charges = np.zeros_like(structure.charges)

    for i, xyz in enumerate(structure.xyz):
        for line in atom_lines:
            if "UNK" in line:
                parts = line.split()
                mol2_xyz = [float(parts[2]), float(parts[3]), float(parts[4])]

                # Check if coordinates match (using np.allclose for float comparison)
                if np.allclose(xyz, mol2_xyz, atol=1e-3):
                    charge = float(parts[-1])
                    new_charges[i] = charge
                    mask[i] = True
                    break  # Move to next atom in structure

    # Update the structure's charges only where mask is True
    structure.charges[mask] = new_charges[mask]

    return structure


def add_charges_to_Structure(combined_id, structure, h5_file_path: str):
    system_id, ligand_id = combined_id.split("/")

    # Extract the ligand
    structure.chain_names = structure.chain_names.astype(str)
    mask = structure.chain_names == ligand_id

    with h5py.File(h5_file_path, "r") as f:
        if "espaloma" in h5_file_path:
            updated_ligand = parse_esp_cif_and_compare(
                f[f"data/{system_id}/{ligand_id}/mol2"][()], structure[mask]
            )
        elif "chargefw2" in h5_file_path:
            updated_ligand = parse_cfw2_cif_and_compare(
                f[f"data/{system_id}/{ligand_id}/cif"][()], structure[mask]
            )
        else:
            raise ValueError("Invalid h5 file path")

    # Create a new structure with the same attributes as the original
    structure.charges[mask] = updated_ligand.charges

    return structure


class StructuresDataset(pt.utils.data.Dataset):
    def __init__(self, pdb_filepaths, rm_wat=False, rm_clash=False):
        super(StructuresDataset).__init__()
        # store dataset filepath
        self.pdb_filepaths = pdb_filepaths

        # store flag
        self.rm_wat = rm_wat
        self.rm_clash = rm_clash

    def __len__(self):
        return len(self.pdb_filepaths)

    def __getitem__(self, i) -> tuple[Structure | None, str]:
        # find pdb filepath
        pdb_filepath = self.pdb_filepaths[i]

        # parse pdb
        structure = load_structure(pdb_filepath)
        return structure, pdb_filepath


class StructuresDatasetPlinder(pt.utils.data.Dataset):
    def __init__(
        self, filepaths, rm_wat=False, rm_clash=False, partial_charges_file=""
    ):
        super(StructuresDataset).__init__()
        self.filepaths = filepaths

        # store flag
        self.rm_wat = rm_wat
        self.rm_clash = rm_clash
        self.partial_charges_file = partial_charges_file

    def __len__(self):
        return len(self.filepaths)


    def __getitem__(self, i) -> tuple[Structure | None, str]:
        filepath = self.filepaths[i]
        try:
            structure = load_structure(filepath)
        except ValueError as e:
            if "unterminated" in str(e):
                filepath = fix_plinder_structure(filepath)
                structure = load_structure(filepath)
            else:
                structure = None
        except Exception:
            structure = None

        if structure is not None and self.partial_charges_file:
            system_id = os.path.dirname(filepath).split("/")[-1]
            for ligand_file in glob(filepath.replace("system.cif", "ligand_files") + "/*.sdf"):
                combined_id = f"{system_id}/{Path(ligand_file).stem}"
                try:
                    structure = add_charges_to_Structure(
                        combined_id, 
                        structure, 
                        self.partial_charges_file
                    )
                except KeyError as e:  # Catch missing HDF5 entries
                    print(f"Failed to add charges for {combined_id}: {str(e)}")
                    structure = None
                    break  # Abort processing this structure entirely

        return structure, filepath

