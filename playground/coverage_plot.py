import numpy as np
import matplotlib.pyplot as plt
import scipy

path = './plots_07_16'
delta_log = np.load(f'{path}/delta_log_finegrid.npy')
percent_levels = [0.005,0.01,0.025,0.05,0.10,0.5,0.90,0.95,0.975,0.99,0.995]
delta_log_levels = [0.0100,0.0201,0.0506,0.1026,0.2107,1.39,4.61,5.99,7.38,9.21,10.60]
coverage = []
print('neg delta log: ', np.sum(delta_log<0))

for i in range(len(percent_levels)):
    coverage.append(np.sum(delta_log < delta_log_levels[i]/2)/len(delta_log))


#print(delta_log)
import matplotlib.pyplot as plt

# plt.figure(figsize=(6, 4))
# plt.plot(percent_levels, coverage, marker='x', markersize = 6, linestyle='-', label='Coverage')

# # Add a dotted y=x line
# plt.plot(percent_levels, percent_levels, 'r--', label='y=x')

# # Add labels and title
# plt.xlabel('Confidence level')
# plt.ylabel('Coverage')
# plt.legend()

# # Add a grid
# plt.grid(True)

# # Show the plot
# plt.show()
# plt.savefig('./plots_11_6/coverage_finegrid_1.pdf')

sorted_delta_log = np.sort(np.delete(delta_log,(delta_log<0)))
chi2 = scipy.stats.chi2.cdf(2*sorted_delta_log,2)

plt.figure()
plt.step(chi2,np.arange(len(sorted_delta_log))/len(sorted_delta_log),where = 'post')
plt.plot(percent_levels, percent_levels, 'r--', label='y=x')
plt.ylabel('Events')
plt.xlabel(r'$\chi^2.cdf(2*\Delta LLH)$')
#plt.legend()

# Add a grid
plt.grid(True)

# Show the plot
plt.show()
plt.savefig(f'{path}/coverage.pdf')
