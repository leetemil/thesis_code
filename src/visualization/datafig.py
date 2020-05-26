import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

uniparcfile = Path('data/files/uniparc.txt')
pdbfile = Path('data/files/pdb.txt')

uniparc = np.loadtxt(uniparcfile, skiprows = 1, dtype = str)
uniparc = {int(d.split('-')[0]) : int(n) for d, n in uniparc}

pdb = np.loadtxt(pdbfile, skiprows = 1, dtype = int)
pdb = {year : n for year, n, _ in pdb}

years = range(2000, 2020)

plt.plot(years, np.array([pdb[year] for year in years]) / 1000000, label = 'Protein Data Bank')
plt.plot(years, np.array([uniparc[year] for year in years]) / 1000000, label = 'UniParc')

plt.title('Protein Database Sizes')
plt.gca().get_xaxis().set_major_locator(matplotlib.ticker.MaxNLocator(integer = True))
plt.gca().get_yaxis().get_major_formatter().set_scientific(False)


plt.xlabel('Year')
plt.ylabel('Size (in millions)')
# plt.yscale('log')
plt.legend()
plt.savefig('../report/figures/protein_size.pdf', bbox_inches='tight')