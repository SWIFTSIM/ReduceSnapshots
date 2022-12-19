#!/usr/bin/env python3

"""
create_empty_reduced_snapshot.py

Create an empty reduced snapshot that looks the same like an actual reduced
snapshot, but with all datasets set to have zero length.
This is required for snapshots that do not contain any halos, since we cannot
produce a working SOAP catalogue for these.

Usage:
  python3 create_empty_reduced_snapshot.py INPUT OUTPUT
where INPUT is the original (full) snapshot and OUTPUT is the name of the empty
snapshot. Note that we only create one empty file, or the equivalent of the
virtual file for a multi-file snapshot,
 i.e. flamingo_0042.hdf5 but not flamingo_0042.1.hdf5
"""

import h5py
import argparse
import os

# dictionary linking human-friendly particle type names and PartTypeX names
partname_dict = {
    "PartType0": "GasParticles",
    "PartType1": "DMParticles",
    "PartType2": "DMBackgroundParticles",
    "PartType3": "SinkParticles",
    "PartType4": "StarsParticles",
    "PartType5": "BHParticles",
    "PartType6": "NeutrinoParticles",
}


class H5copier:
    """
    Functor (class that acts as a function) used to copy datasets and groups
    from one HDF5 file to another.
    """

    def __init__(self, ifile, ofile):
        """
        Constructor. Needs the file from which we copy and the file to which we
        copy.
        """
        self.ifile = ifile
        self.ofile = ofile

    def __call__(self, name, h5obj):
        """
        Function that is called when () is used on an object of this class.
        Conforms to the h5py.Group.visititems() function
        signature.

        Parameters:
         - name: Full path to a group/dataset in the HDF5 file
                  e.g. PartType0/Coordinates
         - h5obj: HDF5 file object pointed at by this path
                   e.g. PartType0/Coordinates --> h5py.Dataset
                        PartType0 --> h5py.Group

        Note that h5py.Group.visititems() completely ignores links in the HDF5
        file, like GasParticls --> PartType0.
        """
        # figure out if we are dealing with a group or a dataset
        type = h5obj.__class__
        if isinstance(h5obj, h5py.Group):
            type = "group"
        elif isinstance(h5obj, h5py.Dataset):
            type = "dataset"
        else:
            raise RuntimeError(f"Unknown HDF5 object type: {name}")

        if type == "group":
            # create a new group with the same name
            self.ofile.create_group(name)
            # copy group attributes, but set all attributes that count numbers
            # of particles to zero, since we want a completely empty snapshot
            for attr in self.ifile[name].attrs:
                if name == "Header" and "NumPart_" in attr:
                    at = self.ifile[name].attrs[attr][:]
                    at[:] = 0
                    self.ofile[name].attrs[attr] = at
                    continue
                if "PartType" in name and attr == "NumberOfParticles":
                    at = self.ifile[name].attrs[attr][:]
                    at[:] = 0
                    self.ofile[name].attrs[attr] = at
                    continue
                self.ofile[name].attrs[attr] = self.ifile[name].attrs[attr]
        elif type == "dataset":
            # copy Cells related datasets:
            #  - keep the Centres as they are
            #  - create full size versions of the other datasets, but set their
            #    elements to 0.
            # In other words: we pretend that the snapshot still uses the same
            # cell structure, but that all the cells are empty.
            if name == "Cells/Centres":
                self.ifile.copy(name, self.ofile, name)
                return
            if "Cells" in name:
                self.ifile.copy(name, self.ofile, name)
                self.ofile[name][:] = 0
                return
            # For all other datasets, we get the dtype and shape and create a
            # new dataset with the same properties but with size (=shape[0]) 0.
            dtype = h5obj.dtype
            shape = h5obj.shape
            new_shape = None
            if len(shape) == 1:
                new_shape = (0,)
            else:
                new_shape = (0, *shape[1:])
            # copy all attributes as they are
            self.ofile.create_dataset(name, new_shape, dtype)
            for attr in self.ifile[name].attrs:
                self.ofile[name].attrs[attr] = self.ifile[name].attrs[attr]


if __name__ == "__main__":
    """
    Main entry point.
    """

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "input", help="Full snapshot file (virtual file or single snapshot file)."
    )
    argparser.add_argument("output", help="Output file.")
    args = argparser.parse_args()

    with h5py.File(args.input, "r") as ifile, h5py.File(args.output, "w") as ofile:
        h5copy = H5copier(ifile, ofile)
        ifile.visititems(h5copy)

        # create soft links, e.g. GasParticles --> PartType0
        for group in partname_dict:
            if group in ofile:
                ofile[partname_dict[group]] = h5py.SoftLink(group)
