import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_pickle('diagnostics.pkl')
df['target_pred'] = 1/(1+np.exp(-np.log(df['target_pred'])))



sns.kdeplot(data=df, x="target_pred", hue="scrambled_class", fill=True, common_norm=False, alpha=0.4, legend = False, clip=(0,1))
plt.yscale('log')
plt.xlim(0,1)
plt.xlabel('Prediction')

plt.tight_layout()
plt.savefig('./plots/diagnostics1.pdf')
plt.show()
plt.close()

sns.kdeplot(data=df, x="target_pred", hue="scrambled_class", multiple='fill', common_norm=False, alpha=0.4, legend = False, clip=(0,1))
plt.xlim(0,1)
plt.xlabel('Prediction')
plt.plot(np.linspace(0,1,2), np.linspace(0,1,2), 'r--', label='y=x')
plt.grid()

plt.tight_layout()
plt.savefig('./plots/diagnostics2.pdf')
plt.show()
plt.close()
