import os
import pandas as pd
import xgboost as xgb
import pickle
from rossmann.Rossmann import Rossmann
import requests
import json
from flask import Flask, request, Response

#loading model
modelo= pickle.load( open( 'model/modelo_rossmann.pkl','rb') )



app = Flask(__name__)
@app.route ('/rossmann/predict', methods = ['POST'] ) #endereço do url (como fosse um www./Methods Post ele envia o dado/é executa função abaixo dele)

def rossmann_predict():
    test_json = request.get_json() #testando a resposta dos dados
    
    if test_json: # there is data
        if isinstance ( test_json, dict ): #unique example
            test_raw = pd.DataFrame ( test_json,index[0] ) #fazendo p/uma linha do Df que contém os daods (unique example)
        
        else: #multiple example
            test_raw = pd.DataFrame ( test_json, columns = test_json[0].keys() ) # coleta várias valores colunas (keys)
        
        # Instantiate Rossmann Class (copiar dados da classe)
        pipeline = Rossmann()
        
        #data cleaning
        
        df1 = pipeline.data_cleaning( test_raw) 
        
        
        #feature_engineering
        
        df2 = pipeline.feature_engineering( df1 )
        
        #  data_preparation
        
        df3 = pipeline.data_preparation( df2 )
        
        #prediction
        df_response = pipeline.get_prediction( modelo, test_raw, df3)
        #model = xgboost; test_raw = df da classe, df3 = dados c/predições
        # carrega o modelo treinado, carrega modelo original e entregue com as predições
        
        return df_response
    
    else:
        return Response ( '{}', status=200, mimetype= 'application/json' )
    

if __name__ == '_main_':
    port = os.environ.get ('PORT',5000)# quando encontrar essa função "main"
    app.run(host= "0.0.0.0", port=port) #método (ligar o carro) / representa que será rodado na máquina local/ sem internet
    