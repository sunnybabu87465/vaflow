import argparse
import os
import shutil

from matplotlib import pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics as ms
import pandas as pd
import numpy as np

def load_testdata_dataframe(csv_path):
    df=pd.read_csv(csv_path,sep=';')
    return df


n_neighbors = 5
test_data_size = .25

parser = argparse.ArgumentParser(
    prog='Vascular access flow prediction')

parser.add_argument('-csv_path', required=False)

colNameMapping = {'Age' : 'Age',
                  'Vintage' : 'Vintage',
                  'Gende' : 'Gender',
                  'Charlson_DiabetesWithOrganDamage' : 'Complicated Diabetes',
                  'ComorbidityDiabetesMellitus' : 'Diabetes',
                  'meanAlbumin' : 'Albumin',
                  'meanCReactiveProtein' : 'C reactive protein',
                  'meanFerritin' : 'Ferritin',
                  'meanGlucose' : 'Glucose',
                  'meanPTH' : 'PTH',
                  'FistulaSurvival' : 'AVF vintage',
                  'CurrentVAPrevFailure' : 'Number of Failures: Current AVF',
                  'LastFailureDistance' : 'Days since Last Failure',
                  'NrPrevClotting' : 'Clotting History',
                  'NrOtherAccessInPeriod' : 'Other active VAs',
                  'LastAccessDistance' : 'Days since the last use of previous vascular access (all types)',
                  'AbsFreqAccess_3M' : 'Number of treatments with AVF: past 3 mo.',
                  'AbsFreqAccess_12M' : 'Number of treatments with AVF: past 12 mo. ',
                  'NrAccess_3M' : 'VA used: past 3 mo.',
                  'NrAccess_6M' : 'VA used: past 6 mo.',
                  'NrFistula_6M' : 'AVF used: past 6 mo.',
                  'NrFistula_12M' : 'AVF used: past 12 mo.',
                  'meanTemperaturePre' : 'Body Temperature',
                  'meanTreatmentSetDuration' : 'Prescribed Treatment Time',
                  'meanTreatmentTime' : 'Treatment Time',
                  'meanUltrafiltrationMaxPerSession' : 'Max Ultrafiltration',
                  'meanEffectiveFlow' : 'Effective flow',
                  'meanTreatmentBloodFlow' : 'Blood flow',
                  'meanTreatmentBloodVolume' : 'Max Processed Blood Volume',
                  'meanDiastolicPre' : 'DBP: pre-dialysis',
                  'meanOCMKTVVA' : 'mean Kt/V',
                  'lastOCMKTVVA' : 'Last Kt/V',
                  'meanRecirculationTRVA' : 'Avg Recirculation',
                  'LastFirstRecirculationTRVA': 'Last Recirculation',
                  'lastVenousPressureMeasuresMaxVA' : 'Dynamic Venous Pressure: Max.',
                  'lastVenousPressureMeasuresMinVA' : 'Dynamic Venous Pressure: Min.',
                  'meanVenousPressureMeasuresMaxDeltaVA' : 'Dynamic Venous Pressure: Max. Variation',
                  'meanVenousPressureMeasuresMeanVA' : 'Dynamic Venous Pressure: Mean',
                  'stdVenousPressureMeasuresMeanVA' : 'Dynamic Venous Pressure: SD',
                  'lastArterialPressureMeasuresMaxVA' : 'Dynamic Arterial Pressure: Max.',
                  'meanArterialPressureMeasuresMaxDeltaVA' : 'Dynamic Arterial Pressure: Max. Variation',
                  'meanArterialPressureMeasuresMeanVA' : 'Dynamic Arterial Pressure: Mean',
                  'stdArterialPressureMeasuresMeanVA' : 'Dynamic Arterial Pressure: SD',
                  'lastBloodFlowMeasuresMaxVA' : 'Qb: Max.',
                  'meanBloodFlowMeasuresMaxDeltaVA' : 'Qb: Max. Variation',
                  'stdBloodFlowMeasuresMeanVA' : 'Average intradialytic blood flow: SD',
                  'sumVReventVA' : 'History of VA complications'
                  }

features = list(colNameMapping.keys())



if __name__ == '__main__':
    patients_df = pd.DataFrame
    args = parser.parse_args()
    if args.csv_path is not None:
        orig_patients_df = load_testdata_dataframe(args.csv_path)
        patients_df = orig_patients_df[features]
    else:
        raise ValueError('Test data source is not provided')

    # fill in the nan values with the mean value of nearest 5 neighbours
    # Data cleaning
    VA_guid = list(orig_patients_df['VascularAccessGUID'].unique())

    df = pd.DataFrame(VA_guid, columns =['VascularAccessGUID'])

    df['train'] = 0

    msk = np.random.rand(len(VA_guid)) < (1-test_data_size)
    df.loc[msk,'train']=1
    df.loc[~msk,'train']=0


    orig_patients_df = pd.merge(orig_patients_df, df, on='VascularAccessGUID', how='left')
    mask = orig_patients_df['train'].values == 1


    Y=orig_patients_df.loc[:,['Qa']]
    Y[orig_patients_df['Qa'] <= 525] = 1
    Y[(orig_patients_df['Qa'] > 525) & (orig_patients_df['Qa'] <= 925)] = 2
    Y[orig_patients_df['Qa'] > 925] = 3

    imputer = KNNImputer(n_neighbors=n_neighbors)
    patients_df_imp = pd.DataFrame(imputer.fit_transform(patients_df), columns=patients_df.columns)


    x_train=patients_df_imp.loc[mask,features]
    x_test=patients_df_imp.loc[~mask,features]
    y_train=Y[mask]
    y_test=Y[~mask]

    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    # # Fit the model on the training data.
    knn.fit(x_train, y_train)

    predictions = knn.predict(x_test).round()
    y_test_int = np.asarray(y_test.values, dtype = 'int')

    print("KNN Accuracy")
    mae = ((abs(predictions - y_test_int)).sum()) / len(predictions)
    print("MAE: %.4f" % mae)
    print(ms.classification_report(y_test_int, predictions, target_names=['very low', 'low', 'normal'], labels=[1, 2, 3]))

    try:
        shutil.rmtree("knn_regression_result")
    except:
        print('Folder not deleted')
    os.mkdir("knn_regression_result")

    cm = ms.confusion_matrix(np.asarray(y_test.values, int), predictions)
    disp = ms.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['very low', 'low', 'normal'])
    disp.plot()
    plt.savefig('knn_regression_result/ConfusionMatrix.png', bbox_inches='tight')
