#!/usr/bin/env python3

"""
test_reduced_snapshot.py

Take a SOAP catalogue and a reduced snapshot and verify that all particles
are present. Not parallelised. Also plots a mass function of the halos kept.

Usage:

  python test_reduced_snapshot.py SOAP_CATALOGUE REDUCED_SNAPSHOT RADIUS

where SOAP_CATALOGUE is the catalogue that was used to generate the
REDUCED_SNAPSHOT, where all particles within RADIUS (e.g. R200c) of
the flagged halos were kept.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import swiftsimio as sw
import unyt

def plot_mass_function(masses_msun):
    fig, ax = plt.subplots(1)
    bin_width = 0.05
    bins = 10**np.arange(13, 16, bin_width)
    mids = (bins[:-1] + bins[1:]) / 2

    n = np.histogram(masses_msun, bins=bins)[0].astype('float64')
    ax.plot(mids[n!= 0], n[n!=0], '.-')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Mass [Msun]')
    ax.set_ylabel(r'$N_{halo}$')
    ax.set_title(f'Bin width: {bin_width} dex')
    plt.savefig('reduced_hmf.png', dpi=200)
    plt.close()


def check_particles_present(soap_filename, snap_filename, radius_name):

    # Load halos, discard those which aren't flagged to be kept
    soap = sw.load(soap_filename)
    soap_mask = soap.soap.included_in_reduced_snapshot == 1
    soap_centre = soap.input_halos.halo_centre[soap_mask]
    if radius_name == 'R200c':
        soap_halo = soap.spherical_overdensity_200_crit
    elif radius_name == 'R100c':
        soap_halo = soap.spherical_overdensity_100_crit
    else:
        raise NotImplementedError
    soap_mass = soap_halo.total_mass[soap_mask]
    soap_r = soap_halo.soradius[soap_mask]
    soap_particle_counts = {
        'dark_matter': soap_halo.number_of_dark_matter_particles[soap_mask]
    }
    try:
        soap_particle_counts['gas'] = soap_halo.number_of_gas_particles[soap_mask]
        soap_particle_counts['stars'] = soap_halo.number_of_star_particles[soap_mask]
    except AttributeError:
        pass

    # Plot mass plot of halos
    plot_mass_function(soap_mass.to('Msun').value)

    # Loop through halos, load all the particles in the surrounding region
    boxsize = soap.metadata.boxsize
    for i_halo in range(soap_r.shape[0]):
        mask = sw.mask(snap_filename)
        load_region = [
            [
                soap_centre[i_halo, i] - 5 * unyt.Mpc,
                soap_centre[i_halo, i] + 5 * unyt.Mpc,
            ] for i in range(3)
        ]
        mask.constrain_spatial(load_region)
        snap = sw.load(snap_filename, mask=mask)
        for part_type, soap_npart in soap_particle_counts.items():
            pos = getattr(snap, part_type).coordinates
            # Set origin to halo centre
            pos = pos + 0.5 * boxsize - soap_centre[i_halo]
            pos %= boxsize
            pos -= 0.5 * boxsize
            r = np.sqrt(np.sum(pos**2, axis=1))
            # Count particles within SO radius
            snap_npart = np.sum(r < soap_r[i_halo])
            tol = 0.0001
            if (snap_npart < (1-tol) * soap_npart[i_halo].value) or (snap_npart > (1+tol) * soap_npart[i_halo].value):
                print(f'Mismatch in number of {part_type} particles for halo {i_halo}')
                print(f'Snapshot: {snap_npart}, SOAP: {soap_npart[i_halo].value}')


if __name__ == "__main__":
    """
    Main entry point.
    """

    argparser = argparse.ArgumentParser()
    argparser.add_argument("soap", help="SOAP catalogue")
    argparser.add_argument("snap", help="Reduced snapshot")
    argparser.add_argument(
        "radius", help="Radius within which particles are kept. Valid options: R200c, R100c"
    )
    args = argparser.parse_args()

    check_particles_present(args.soap, args.snap, args.radius)
