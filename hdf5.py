import numpy as np
import h5py

def h5_load(archive, section):
    data = archive[section].value
    if "__complex__" in archive[section].attrs:
        data = data[...,0] + 1.0j * data[...,1]
    return data

def h5_save(archive, section, data):
    if np.iscomplexobj(data):
        archive.create_dataset(section, data = np.stack([data.real, data.imag], axis=-1))
        archive[section].attrs.create("__complex__", 1)
    else:
        archive.create_dataset(section, data = data)

def load_grid_object(archive, section):
    data = h5_load(archive, section + "/data")

    grids = []
    for i, g in archive[section + "/grids/"].items():
        grids += [g.value]

    return data, grids

def save_grid_object(archive, section, data, grids):
    assert(len(grids) == len(data.shape))
    h5_save(archive, section + "/data", data)
    for i, g in enumerate(grids):
        h5_save(archive, section + "/grids/{}".format(i), g)

def load_green(archive, section):
    data, grids = load_grid_object(archive, section + "/gtr_up")

    t0 = grids[0]
    t1 = grids[1]

    Green = np.zeros((2, 2, len(t0), len(t0)), complex)

    Green[0,0,:,:] = data

    data, grids = load_grid_object(archive, section + "/les_up")
    Green[1,0,:,:] = data

    data, grids = load_grid_object(archive, section + "/gtr_down")
    Green[0,1,:,:] = data

    data, grids = load_grid_object(archive, section + "/les_down")
    Green[1,1,:,:] = data

    return Green, grids

def save_green(archive, section, Green, grids):
    #indices grt/less, up/down, time, times
    save_grid_object(archive, section + "/gtr_up",   Green[0,0,:,:], grids)
    save_grid_object(archive, section + "/les_up",   Green[1,0,:,:], grids)
    save_grid_object(archive, section + "/gtr_down", Green[0,1,:,:], grids)
    save_grid_object(archive, section + "/les_down", Green[1,1,:,:], grids)
