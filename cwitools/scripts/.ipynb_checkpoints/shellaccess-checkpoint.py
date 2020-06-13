import cwitools.scripts as scripts

def apply_mask(mask=None, data=None, fill=0, ext=".M.fits", log=None,
    silent=False, label=0):
    """Wrapper for applying mask to data - Docstring TBC"""
    scripts.apply_mask.main(mask, data, fill, ext, log, silent, label)

cwi_measurewcs = scripts.measurewcs.main