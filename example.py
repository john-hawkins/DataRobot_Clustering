# Import the file: drclustering.py
import drclustering as drc
import datarobot as dr
import pandas as pd

# DEFINE THE PROJECT AND MODEL WE WANT TO USE
PROJECT_ID = '5b63acf20c609e426492fecc'
MODEL_ID = '5b63afa50b701902c9747ae5'

# LOAD THE PROJECT AND MODEL
proj = dr.Project.get(project_id=PROJECT_ID)
mod =  dr.Model.get(PROJECT_ID, model_id=MODEL_ID)

# LOAD THE SAMPLE DATA THAT WILL BE USED
data = pd.read_csv('./test.csv')

# DEFINE THE TWO COLUMNS WE WILL USE
colone = "NUM_PI_CLAIM"
coltwo = "DISTINCT_PARTIES_ON_CLAIM"

# NUMBER OF CLUSTERS TO CREATE
kvalue = 5

######################################################################################
# Create the plot and show it
######################################################################################

plt = drc.generate_kmeans_cluster_plot(proj, mod, data, kvalue, colone, coltwo) 
plt.show()

