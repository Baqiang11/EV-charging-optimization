import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#Create graphics and coordinate axes
fig, ax = plt.subplots(figsize=(10, 6))


ax.text(0.5, 0.9, 'Random Forest Model', fontsize=14, ha='center', va='center', bbox=dict(facecolor='lightblue', edgecolor='black'))
ax.text(0.2, 0.7, 'Training Data', fontsize=12, ha='center', va='center', bbox=dict(facecolor='lightyellow', edgecolor='black'))
ax.text(0.8, 0.7, 'Multiple Decision Trees', fontsize=12, ha='center', va='center', bbox=dict(facecolor='lightgreen', edgecolor='black'))
ax.text(0.5, 0.5, 'Bootstrap Sampling', fontsize=12, ha='center', va='center', bbox=dict(facecolor='lightgray', edgecolor='black'))
ax.text(0.5, 0.3, 'Aggregate Results (Voting/ Averaging)', fontsize=12, ha='center', va='center', bbox=dict(facecolor='lightcoral', edgecolor='black'))
ax.text(0.5, 0.1, 'Final Prediction', fontsize=12, ha='center', va='center', bbox=dict(facecolor='lightblue', edgecolor='black'))


ax.arrow(0.5, 0.85, 0, -0.05, head_width=0.03, head_length=0.02, fc='black', ec='black')
ax.arrow(0.2, 0.65, 0.3, -0.1, head_width=0.03, head_length=0.02, fc='black', ec='black')
ax.arrow(0.8, 0.65, -0.3, -0.1, head_width=0.03, head_length=0.02, fc='black', ec='black')
ax.arrow(0.5, 0.45, 0, -0.05, head_width=0.03, head_length=0.02, fc='black', ec='black')
ax.arrow(0.5, 0.25, 0, -0.05, head_width=0.03, head_length=0.02, fc='black', ec='black')


ax.axis('off')


plt.show()
