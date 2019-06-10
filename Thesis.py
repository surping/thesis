import pandas as pd

hsptl = pd.read_csv('https://raw.githubusercontent.com/HoloClean/holoclean/master/testdata/hospital.csv')
hsptl_cln = pd.read_csv('https://raw.githubusercontent.com/HoloClean/holoclean/master/testdata/hospital_clean.csv')

hsptl_cln = pd.pivot_table(hsptl_cln, index='tid', columns='attribute', values='correct_val', aggfunc='first', dropna=False)
hsptl_cln['Address2'] = hsptl_cln.Address2.astype(float)
hsptl_cln['Address3'] = hsptl_cln.Address3.astype(float)

new_df = pd.merge(hsptl, hsptl_cln,  how='inner', on = ['ProviderNumber','HospitalName','Address1','Address2','Address3','City','State','ZipCode','CountyName','PhoneNumber','HospitalType','HospitalOwner','EmergencyService','Condition','MeasureCode','MeasureName','Score','Sample','Stateavg'])

print(new_df)


# hsptl_cln.dtypes