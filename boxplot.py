
import pandas as pd
import matplotlib.pyplot as plt
#%%
data = pd.read_excel("./Pirority实验.xlsx", sheet_name=1)


#%%
data.plot.box()
plt.tick_params(labelsize=15)
plt.grid(linestyle="--", alpha=0.3)
plt.savefig("/data0/jinhaibo/lixiaohao/priority/figure/TOP100acc.png", dpi=300)
plt.show()

#%%
data[["LSA", "DSA", "PRIMA", "DeepGini", "BallPri"]].boxplot(by="LSA")
plt.xlabel("横坐标XXX")
plt.ylabel("纵坐标XXX")
plt.title("箱式图")
plt.show()