import matplotlib.pyplot as plt
import numpy as np

offnight_cals = np.loadtxt('/home/donal/data/flashes/meta/pilot_offnight_cals.txt',
                                dtype=str
)

nebular_data = np.genfromtxt('/home/donal/data/flashes/meta/nebularProperties_snr1.txt',
                                names = ('target','reff')
                                usecols = (0,2),
                                skipheader = True,
                                dtype = None
)
