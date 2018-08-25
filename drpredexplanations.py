# #############################################################################
# Methods for retriveing and manipulating the DataRobot prediction explanations
#  for a new dataset
# #############################################################################
from functools import reduce
import datarobot as dr
import pandas as pd

# ######################################################################
# UTILITY FUNCTIONS FOR CLEAN RE_USABLE BEHAVIOUR
# ######################################################################
def unlist(listOfLists):
    return [item for sublist in listOfLists for item in sublist]

def unique_elements(bigList):
    return reduce(lambda l, x: l.append(x) or l if x not in l else l, bigList, [])


# #####################################################################
# Generic method to do everything required to retrieve the scores and
# explanations for a given data set.
# #####################################################################
def retrieve_prediction_explanations(proj, mod, pdata, n_reasons = 5):
    # UPLOAD THE DATASET
    dataset = proj.upload_dataset(pdata) # Returns an instance of [PredictionDataset]
    pred_job = mod.request_predictions(dataset.id)
    preds = pred_job.get_result_when_complete()
    # NOW WE NEED TO ENSURE THAT FEATURE IMPACT EXISTS FOR THAT MODEL
    try:
        impact_job = mod.request_feature_impact()
        impact_job.wait_for_completion(300)
    except dr.errors.JobAlreadyRequested:
        pass  # already computed
    # NOW ENSURE THAT THE PREDICTION EXPLANATIONS ARE COMPUTED 
    try:
        dr.ReasonCodesInitialization.get(proj.id, mod.id)
    except dr.errors.ClientError as e:
        assert e.status_code == 404  # haven't been computed
        init_job = dr.ReasonCodesInitialization.create(proj.id, mod.id)
        init_job.wait_for_completion()
    # RUN THE REASON CODE JOB
    rc_job = dr.ReasonCodes.create(proj.id,
                               mod.id,
                               dataset.id,
                               max_codes=n_reasons,
                               threshold_low=None,
                               threshold_high=None)
    rc = rc_job.get_result_when_complete(max_wait=1200)
    all_rows = rc.get_all_as_dataframe()
    return all_rows


# #############################################################
# TRANSFORMATION OF THE PREDICTION EXPLANATIONS INTO A SET OF 
# COLUMNS PER FEATURE WITH THE QUANTITATIVE PREDICTION STRENGTH 
# VALUE IN THE DATA CELL
# WE NEED TO KNOW THE PROJECT TYPE TO DETERMINE THE COLUMN NUMBER
# WHERE THE EXPLANATIONS START.
# #############################################################
def get_strength_per_feature_cols(proj, all_rows, n_reasons=5):
    colsToUse = []
    startPoint = 6
    if proj.target_type == 'Regression':
        startPoint = 2
    if proj.target_type == 'Binary':
        startPoint = 6
    j = startPoint
    for i in range(n_reasons):
        colsToUse.append(j) 
        colsToUse.append(j+4)
        j = j + 5 
    rc3 = all_rows.iloc[:, colsToUse]
    j = 0
    colsForNames = []
    for i in range(n_reasons):
        colsForNames.append(j) 
        j = j + 2 
    namesdf = rc3.iloc[:,colsForNames]
    allfeatures = [namesdf[i].unique().tolist() for i in namesdf.columns]
    nameslist = unique_elements(unlist(allfeatures))
    ####################################################################
    # CREATE A NEW DATAFRAME WITH ONE COLUMN PER POSSIBLE REASON CODE
    # INITIALISE TO ZERO AND THEN FILL WITH THE EXPLANATION STRENGTHS
    ####################################################################
    dfnew = pd.DataFrame(columns=nameslist)
    for j in range(len(rc3)):
        dfnew.loc[j] = [0 for n in range(len(nameslist))]
        for i in range(n_reasons):
            rcname = rc3.loc[j][i*2] 
            rcvalue = rc3.loc[j][i*2+1]
            dfnew.loc[j][rcname] = rcvalue    
    return dfnew    

