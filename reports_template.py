# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 14:33:08 2023

@author: o_hol
"""

#REPORTS TEMPLATE


from dataclasses import dataclass
import pandas as pd
import pyodbc
from ydata_profiling import ProfileReport
import urllib


@dataclass
#dataprep, autoviz and lux not implemented
class Reports:
    #df
    
    def create_profiling_report(self,df,title="Data Profile Report",**kwargs):
        """
        Create automatic report from pandas dataframe or spark dataframe
        """
        profile = ProfileReport(df,title=title,**kwargs)
        #profile.to_file("report.html")
        #profile.to_html()
        #from profile we can use profile.to_file("filename.html") and profile.to_html()
        return profile
    
    def dataprep_report(self,df,**kwargs): #PENDING
        from dataprep.eda import create_report
        report = create_report(df)
        return report
    
    def retro_report(self,df,**kwargs):
        compare = kwargs.get("compare")
        #if compare implement logic
        import sweetviz as sv
        rp2 = sv.analyze(df)
        #rp2.show_html()
        return rp2
        
        
        
        
#RUNNING EXAMPLE FROM SQL DB TABLE

if __name__ == "main":
    sqlServer = {'sqlServerName': 'euwdsrg03rsql01.database.windows.net',
     'sqlDatabase': 'EUWDSRG03RRSG02ADB01_Copy',
     'userName': 'dbWSS',
     'password': 'braf0wNVtixu3?IhU=hmrCeLzmzX>Wlo'}



    # PROD_MxD_PDM_DeviceFailureV2DataTable
    sqlServerName,sqlDatabase,userName,password = sqlServer.get('sqlServerName'),sqlServer.get('sqlDatabase'),sqlServer.get('userName'), sqlServer.get('password')


    def read_sql(sqlServerName ,sqlDatabase,userName,password,tablename,sqlPort = 1433,query =None,pandas=False)->pd.DataFrame:
        query = query if query else f"(SELECT * FROM {tablename})"# AS subquery"
        try:
            if pandas:
                cnxn = pyodbc.connect(DRIVER="{ODBC Driver 17 for SQL Server}", SERVER=sqlServerName, DATABASE=sqlDatabase, UID=userName, PWD=password, STORE_DRVRESULTS=0)
                # cursor = cnxn.cursor()
                # query = f"SELECT * FROM {tablename}"
                # cursor.execute(query)
                # columns = [row.column_name for row in cursor.columns(table=tablename)]
                # df=
                df = pd.read_sql(query,cnxn)
                return df
            #df.write.jdbc(sqlServerUrl, "PDM_AD_PredictionTable", write_mode, connectionProperties)
            else:
                connstr = f"DRIVER=ODBC Driver 17 for SQL Server, SERVER={sqlServerName}, DATABASE={sqlDatabase}, UID={userName}, PWD={password}, STORE_DRVRESULTS=0"
                connection_string = urllib.parse.quote_plus(connstr)
                connection_string = "mssql+pyodbc:///?odbc_connect=%s" % connection_string
                return connection_string
        except Exception as e:
            return str(e)


    sqldf = lambda table: read_sql(sqlServerName ,sqlDatabase,userName,password,table,sqlPort = 1433,pandas = True)
    tables = ["PROD_MxD_PDM_DeviceFailureV2DataTable","PROD_MxD_PDM_DeviceFailureV2PredictionTable","PROD_MxD_DDM_AssetDataTable","PROD_MxD_DDM_DowntimeDataTable"]

    dfs=[sqldf(table) for table in tables]
    
    #REPORTS
    generic_report = Reports()
    pdf1 = dfs[0][["WindowTimeStamp_Start","WindowTimeStamp_End","DeviceName","Vibration","Voltage","AirPressure","Current_amps"]]
    r1 = generic_report.create_profiling_rep
        
