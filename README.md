This directory contains the software used to generate reduced snapshots.
This software is written in C++ and needs to be compiled using the provided
compilation script.

The reduction software will:
 1. Read the SOAP catalogue and create a list of halos that need to be kept.
 2. Loop over the cells in the SWIFT snapshot and for each cell, find particles
    that are inside a halo that needs to be kept. It will also assign a halo ID
    to each of these particles; if a halo belongs to multiple halos, the ID of
    the most massive halo is kept.
 3. All the datasets for particles with a valid halo ID are copied into a new
    snapshot file, filtering out all other particles.
 4. The particle number metadata is synced across the new snapshot files.
 5. A new virtual snapshot file is produced for the new snapshot files.

Example job scripts for various FLAMINGO simulations are provided.

Some snapshots do not contain any halos. Reduced snapshots for these snapshots
are trivial, but can be hard to produce, since SOAP does not run on these
snapshots. `create_empty_reduced_snapshot.py` can be used to generate empty
reduced snapshots for those: snapshots that have the same structure as the
other reduced snapshots, but that only contain size 0 datasets. This makes it
possible to blindly run scripts on all reduced snapshots, even the empty ones.
