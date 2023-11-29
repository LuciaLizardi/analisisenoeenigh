
import pandas as pd

#Trimestres 2023
enoe_sdem23t1=pd.read_csv('E:/Users/lucia/ProyectoInferencia/2023/ENOE_SDEMT123.csv')
enoe_sdem23t2=pd.read_csv('E:/Users/lucia/ProyectoInferencia/2023/ENOE_SDEMT223.csv')
#Trimestres 2022
enoe_sdem22t1=pd.read_csv('E:/Users/lucia/ProyectoInferencia/2022/ENOEN_SDEMT122.csv')
enoe_sdem22t2=pd.read_csv('E:/Users/lucia/ProyectoInferencia/2022/ENOEN_SDEMT222.csv')
enoe_sdem22t3=pd.read_csv('E:/Users/lucia/ProyectoInferencia/2022/ENOEN_SDEMT322.csv')
enoe_sdem22t4=pd.read_csv('E:/Users/lucia/ProyectoInferencia/2022/ENOEN_SDEMT422.csv')
#Trimestres 2021
'''enoe_sdem21t1=pd.read_csv('E:/Users/lucia/ProyectoInferencia/2021/ENOEN_SDEMT121.csv')
enoe_sdem21t2=pd.read_csv('E:/Users/lucia/ProyectoInferencia/2021/ENOEN_SDEMT221.csv')
enoe_sdem21t3=pd.read_csv('E:/Users/lucia/ProyectoInferencia/2021/ENOEN_SDEMT321.csv')
enoe_sdem21t4=pd.read_csv('E:/Users/lucia/ProyectoInferencia/2021/ENOEN_SDEMT421.csv')'''
#Trimestres 2019
enoe_sdem19t1=pd.read_csv('E:/Users/lucia/ProyectoInferencia/2019/sdemt119.csv')
enoe_sdem19t2=pd.read_csv('E:/Users/lucia/ProyectoInferencia/2019/sdemt219.csv')
enoe_sdem19t3=pd.read_csv('E:/Users/lucia/ProyectoInferencia/2019/sdemt319.csv')
enoe_sdem19t4=pd.read_csv('E:/Users/lucia/ProyectoInferencia/2019/sdemt419.csv')

#2023
datos_23=enoe_sdem23t1.append(enoe_sdem23t2, ignore_index=True)
#2022
datos_22=enoe_sdem22t1.append(enoe_sdem22t2, ignore_index=True)
datos_22=datos_22.append(enoe_sdem22t3, ignore_index=True)
datos_22=datos_22.append(enoe_sdem22t4, ignore_index=True)
#2019
datos_19=enoe_sdem19t1.append(enoe_sdem19t2, ignore_index=True)
datos_19=datos_19.append(enoe_sdem19t3, ignore_index=True)
datos_19=datos_19.append(enoe_sdem19t4, ignore_index=True)        

#Limpiando 
#Defino mis variables de interés AKA - Nombres de las columnas
var_interes=['cd_a','per','sex','eda','cs_p12','cs_p13_1','clase1','clase2','pos_ocu'
             ,'seg_soc','rama','c_ocu11c','ing7c','dur9c','medica5c','rama_est1'
             ,'rama_est2','dur_est']   
#Filtro solo esas columnas por años
enoe_23fltr=datos_23.loc[:, var_interes]
enoe_22fltr=datos_22.loc[:, var_interes]
enoe_19fltr=datos_19.loc[:, var_interes]

#Dataframe final con los 3 años y solo las columnas de interes
final_enoe=enoe_23fltr.append(enoe_22fltr, ignore_index=True)
final_enoe=final_enoe.append(enoe_19fltr, ignore_index=True)


final_enoe.to_csv('E:/Users/lucia/ProyectoInferencia/enoe_limpia19a23.csv', index=False)
