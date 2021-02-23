    import pandas as pd
    
    #hospitals process start
    hsptl = pd.read_csv('https://raw.githubusercontent.com/HoloClean/holoclean/master/testdata/hospital.csv')
    hsptl_cln = pd.read_csv('https://raw.githubusercontent.com/HoloClean/holoclean/master/testdata/hospital_clean.csv')

    hsptl_cln = pd.pivot_table(hsptl_cln, index='tid', columns='attribute', values='correct_val', aggfunc='first', dropna=False)
    hsptl_cln['Address2'] = hsptl_cln.Address2.astype(float)
    hsptl_cln['Address3'] = hsptl_cln.Address3.astype(float)

    #Find the hospitals with the correct values and flag them
    correct_hsptl = pd.merge(hsptl, hsptl_cln,  how='inner', on = ['ProviderNumber','HospitalName','Address1','Address2','Address3','City','State','ZipCode','CountyName','PhoneNumber','HospitalType','HospitalOwner','EmergencyService','Condition','MeasureCode','MeasureName','Score','Sample','Stateavg'])
    correct_hsptl['correct_flg'] = '1'

    #Merge the dataframe with all hospitals and the dataframe which has the correct flaging in order to apply the flagging on the main dataframe
    hsptl = pd.merge(hsptl, correct_hsptl,  how='left', on = ['ProviderNumber','HospitalName','Address1','Address2','Address3','City','State','ZipCode','CountyName','PhoneNumber','HospitalType','HospitalOwner','EmergencyService','Condition','MeasureCode','MeasureName','Score','Sample','Stateavg'])   
    #hospitals process end

    #flights process start
    flights = pd.read_csv('https://raw.githubusercontent.com/HoloClean/holoclean/master/testdata/flight.csv')
    flights_cln = pd.read_csv('https://raw.githubusercontent.com/HoloClean/holoclean/master/testdata/flight_clean.csv')

    flights_cln = pd.pivot_table(flights_cln, index='tid', columns='attribute', values='correct_val', aggfunc='first', dropna=False).reset_index(['tid'])
    flights_cln.columns = ['flight','actual_arrival','actual_dept','arrival_gate','dept_gate','scheduled_arrival','scheduled_dept']

    #Find the flights with the correct values and flag them
    correct_flght = pd.merge(flights, flights_cln,  how='inner', on = ['flight','scheduled_dept','actual_dept','dept_gate','scheduled_arrival','actual_arrival','arrival_gate'])
    correct_flght['correct_flg'] = '1'

    #Merge the dataframe with all flights and the dataframe which has the correct flaging in order to apply the flagging on the main dataframe
    flights = pd.merge(flights, correct_flght,  how='left', on = ['flight','scheduled_dept','actual_dept','dept_gate','scheduled_arrival','actual_arrival','arrival_gate', 'src'])    
    #flights process end

    export_csv = hsptl.to_csv (r'C:\Users\d-nmoutsopoulos\Documents\Projects\Thesis\hospitals.csv', index = None, header=True)
    export_csv = flights.to_csv (r'C:\Users\d-nmoutsopoulos\Documents\Projects\Thesis\flights.csv', index = None, header=True)
