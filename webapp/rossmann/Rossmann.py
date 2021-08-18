import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime

# ex. carro (objeto) 
class Rossmann ( object ): #limpeza, transformação e encoding (para usar no api)/ nessa etapa utiliza-se esssa classe para transformar novas entradas
    
    def __init__( self ): # _init_ é usado para ativar/inicar as propriedades(valores) da classe
        self.home_path= '' #caminho absoluto/raiz da pasta
        self.competition_distance_scaler   = pickle.load( open( self.home_path + 'parameter/competition_distance_scaler.pkl','rb') )
        self.competition_time_month_scaler = pickle.load( open( self.home_path + 'parameter/competition_time_month_scaler.pkl','rb') )
        self.promo_time_week_scaler        = pickle.load( open( self.home_path + 'parameter/promo_time_week_scaler.pkl','rb') )
        self.year_scaler                   = pickle.load( open( self.home_path + 'parameter/year_scaler.pkl','rb') )
        self.store_type_scaler             = pickle.load( open( self.home_path + 'parameter/store_type_scaler.pkl','rb') )
        
    def data_cleaning ( self, df1 ): #self (si mesmo)/ usado para referenciar o objeto que irá herdas as propriedas(valores)
       
        # 1.1 Rename columns # foram retiradas as colunas Sales e Clustomers
        cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 
                    'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth',
                    'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']


        snackcase= lambda x: inflection.underscore( x )
    
        new_cols = list(map(snackcase,cols_old))
        
        #rename columns 
        df1.columns = new_cols
        
        ## 1.3 Data Type
        df1['date'] =pd.to_datetime(df1['date'])
        
        ## 1.5 Fillout NA
        # competition_distance
        df1['competition_distance']= df1['competition_distance'].apply(lambda x: 200000.0 if math.isnan(x) else x )

         
         # competition_open_since_month   
        df1['competition_open_since_month'] = df1.apply(lambda x: x['date'].month if math.isnan(x['competition_open_since_month']) else x['competition_open_since_month'], axis=1)

        ## competition_open_since_year
        df1['competition_open_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(x['competition_open_since_year']) else x['competition_open_since_year'], axis=1)

        ## promo2_since_week 
        df1['promo2_since_week']= df1.apply(lambda x: x['date'].week if math.isnan(x['promo2_since_week']) else x['promo2_since_week'],axis=1)
                                    
        ## promo2_since_year
        df1['promo2_since_year'] = df1.apply(lambda x:x['date'].year if math.isnan(x['promo2_since_year'])else x['promo2_since_year'],axis=1)
             
        ## month_map
        month_map = {1: 'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}

        ## promo_interval
        df1['promo_interval'].fillna(0, inplace=True)

        ## month_map  (renomeando todo DF para os meses numericamente)
        df1['month_map']= df1['date'].dt.month.map(month_map)

        # is_promo
        df1['is_promo'] = df1[['promo_interval','month_map']].apply(lambda x: 0 if x['promo_interval'] ==0 else 1 if x['month_map'] in x['promo_interval'].split(',') else 0,axis=1)

        #1.6 Change Data Types
        #competition
        df1['competition_distance']= df1['competition_distance'].astype('int64')
        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype('int64')
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype('int64')
        
        #promo2
        df1['promo2_since_week'] = df1['promo2_since_week'].astype( int )
        df1['promo2_since_year'] = df1['promo2_since_year'].astype( int )
        

        return df1

    def feature_engineering( self, df2 ):
        
        #year    
        df2['year']= df2['date'].dt.year
        
        #month
        df2['month']= df2['date'].dt.month
        
        # day
        df2['day']= df2['date'].dt.day
        
        # week of year
        df2['week_of_year']= df2['date'].dt.weekofyear
        #
        # year week (semana desse ano)
        df2['year_week']= df2['date'].dt.strftime('%Y-%W') #coluna criada para formatar data como"ano e semana desse ano".

        #competition_since

        # competition_since
        df2['competition_since'] = df2.apply( lambda x: datetime.datetime( year=x['competition_open_since_year'], month=x['competition_open_since_month'],day=1 ), axis=1 )

        ##tempo de competição em meses. (subtraindo  coluna de datas com coluna competition_since que foi transformada tbm em data)
        df2['competition_time_month']= ((df2['date'] - df2['competition_since'])/30).apply(lambda x: x.days).astype( int)

        #promo_since

        #primeiro deve transformar as duas colunas em strings para unir as datas
        df2['promo_since']= df2['promo2_since_year'].astype( str) + '-' + df2['promo2_since_week'].astype( str )
        df2['promo_since']= df2['promo_since'].apply(lambda x: datetime.datetime.strptime( x + '-1','%Y-%W-%w') - datetime.timedelta(days=7))
        df2['promo_time_week'] = ((df2['date'] - df2['promo_since'])/7).apply(lambda x: x.days).astype (int )

        ##assortment (a = basic, b = extra, c = extended) - objetvo aqui é mudar letras para descrição completa no DF.
        
        df2['assortment'] = df2['assortment'].apply(lambda x: 'basic' if x== 'a' else 'extra' if x =='b'else 'extended')

        #State_holiday a = public holiday, b = Easter holiday, c = Christmas, 0 = None

        df2['state_holiday']= df2['state_holiday'].apply(lambda x: 'public_holiday' if x=='a' else 'easter_holiday' if x=='b' else 'christmas' if x=='c' else 'regular_day')
        
        
        # filtrando as colunas (lojas abertas com vendas maiores que 0)
        df2 = df2[df2['open'] != 0] # o filtro foi retirado os sales pois não há mais necessidade

        # selecionando as colunas que importam (removendo com método drop)
        cols_drop= ['month_map','promo_interval','open']
        df2 = df2.drop( cols_drop, axis=1 )
       
        
        return df2
        
    def data_preparation( self, df5):
                
        # 5.3 Rescaling
        #competition_distance robust scaller
        df5['competition_distance'] = self.competition_distance_scaler.fit_transform(df5[['competition_distance']].values)
 
        #competition_time_month robust scaller
        df5['competition_time_month'] = self.competition_time_month_scaler.fit_transform(df5[['competition_time_month']].values)
                  
        # promo_time_week minimo scaller
        df5['promo_time_week'] = self.promo_time_week_scaler.fit_transform(df5[['promo_time_week']].values)

        # year competition minimo scaller                                                                
        df5['year'] = self.year_scaler.fit_transform(df5[['year']].values)

        # 5.3.1 Enconding
        # One hot = Pois os feriados são um estado (momento). Esse tipo de enconder é excelente para essa ocasião. 
        # state_holiday = one hot
        df5 = pd.get_dummies( df5, prefix =['state_holiday'], columns =['state_holiday'])
 
        # store_type = label enconding
        df5['store_type'] = self.store_type_scaler.fit_transform( df5['store_type'] )
        
        # dict_assorment = ordinal enconding (coloca as encoding em hieraquia)
        dict_assortment = {'basic': 1, 'extra': 2, 'extended': 3}
        df5['assortment'] = df5['assortment'].map(dict_assortment)
         
        # 5.3.1 Nature transformation
        # day
        df5['day_sin'] = df5['day'].apply(lambda x: np.sin( x * ( 2. * np.pi/30 ) ) )
        df5['day_cos'] = df5['day'].apply(lambda x: np.cos( x * ( 2. * np.pi/30 ) ) )


        #month
        df5['month_sin'] = df5['month'].apply( lambda x: np.sin( x *( 2 * np.pi/12 ) ) )
        df5['month_cos'] = df5['month'].apply( lambda x: np.cos( x * (2 * np.pi/12 ) ) )

        # day_of_week

        df5['day_of_week_sin'] = df5['day_of_week'].apply( lambda x: np.sin( x  * (2. * np.pi/7 ) ) ) 
        df5['day_of_week_cos'] = df5['day_of_week'].apply( lambda x: np.cos( x  * (2. * np.pi/7 ) ) )


        #week_of_year
        df5['week_of_year_sin'] = df5['week_of_year'].apply(lambda x: np.sin( x * ( 2. * np.pi/52 ) ) )
        df5['week_of_year_cos'] = df5['week_of_year'].apply(lambda x: np.cos( x * ( 2. * np.pi/52 ) ) )
        
        
        cols_selected=['store','promo','store_type','assortment','competition_distance','competition_open_since_month',
            'competition_open_since_year','promo2','promo2_since_week','promo2_since_year','competition_time_month','promo_time_week',
             'day_of_week_sin','day_of_week_cos','month_sin','month_cos','day_sin','day_cos','week_of_year_sin','week_of_year_cos']
                        
        
        return df5[cols_selected]
        
    def get_prediction( self, modelo, original_data, test_data):
                  #prediction
        pred = modelo.predict( test_data )
                  
        #join pred into the original data
        original_data['prediction'] = np.expm1( pred ) #transformando expoente
            
        return original_data.to_json(orient = 'records', date_format = 'iso')
                  #retorne a váriavel original_data e transforme em um dict json (iso = não bagunçar datas)