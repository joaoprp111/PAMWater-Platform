from flask import Flask, jsonify, Response, request, after_this_request
import mysql.connector
import pandas as pd
import json
import joblib
from sklearn.tree import _tree
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from flask import send_file
from datetime import datetime
from tensorflow.keras.models import model_from_json

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'HELLO WORLD'


mydb = mysql.connector.connect(
    host="mysql-95018-0.cloudclusters.net",
    port=19044,
    user="admin",
    passwd="IlzUqWhz",
    database="pamwater"
)

cursor = mydb.cursor()

query_indicator_electricidade = "SELECT indicator_table.indicator_name, indicator_table.indicator_type, indicator_table.units, indicator_value_table.sub_type, indicator_value_table.input, indicator_value_table.value, indicator_value_table.date, indicator_value_table.city_name FROM indicator_table INNER JOIN indicator_value_table ON indicator_table.id = indicator_value_table.indicator"
cursor.execute(query_indicator_electricidade)
result_indicator_electricidade = cursor.fetchall()

data_db = pd.DataFrame((result_indicator_electricidade),columns=['indicator_name','indicator_type','units','sub_type','input','value','date','city_name'])
data_db_elec = data_db[data_db.indicator_type == 'Electricidade']
data_db = data_db[data_db.indicator_type == 'Controlo Analitico']

data = data_db.loc[((data_db.sub_type == 'Afluente Bruto') | (data_db.sub_type == 'Efluente Tratado'))]
data.date = pd.to_datetime(data.date).dt.date
data.date = pd.to_datetime(data.date)

data_ph = data_db.loc[(data_db.sub_type == 'Afluente Bruto') & (data_db.indicator_name == 'ph')]
data_ph.date = pd.to_datetime(data_ph.date).dt.date
data_ph.date = pd.to_datetime(data_ph.date)

data_elec = data_db_elec.loc[(data_db_elec.indicator_name == 'total') & (data_db_elec.city_name == 'Guimaraes')]
data_elec.date = pd.to_datetime(data_elec.date).dt.date
data_elec.date = pd.to_datetime(data_elec.date)

def series_to_supervised(data, timesteps, multisteps, dropnan=False, fill_value=0):
    data = pd.DataFrame(data)
    new = pd.DataFrame()
    for i in range(timesteps, 0, -1):
        if fill_value:
            new = pd.concat([new, data.shift(i, fill_value=fill_value)], axis=1)
        else:
            new = pd.concat([new, data.shift(i)], axis=1)

    for j in range(0, multisteps):
        if fill_value:
            new = pd.concat([new, data.iloc[:,0].shift(-j, fill_value=fill_value)],axis=1)
        else:
            new = pd.concat([new, data.iloc[:,0].shift(-j)],axis=1)
    if dropnan:
        new.dropna(inplace=True)
    return new.values

@app.route('/dados')
def dados():
    @after_this_request
    def add_header(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    #result = data.to_json(orient="records")
    return Response(data.to_json(orient="records"), mimetype='application/json')


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    @after_this_request
    def add_header(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    if request.method == 'POST':
        print(request.form)
        #result_post = request.form
        pred = [[request.form['azoto_total_em_Afluente_Bruto'], request.form['cqo_em_Efluente_Tratado'],
                 request.form['sst_em_Afluente_Bruto'], request.form['amonia_em_Efluente_Tratado'],	request.form['ortofosfatos_em_Efluente_Tratado']]]
    loaded_model = joblib.load('dt_model.sav')
    prediction = loaded_model.predict(pred)
    result = {'azoto_total_em_Afluente_Bruto': request.form['azoto_total_em_Afluente_Bruto'], 'cqo_em_Efluente_Tratado': request.form['cqo_em_Efluente_Tratado'], 'sst_em_Afluente_Bruto': request.form['sst_em_Afluente_Bruto'],
              'amonia_em_Efluente_Tratado': request.form['amonia_em_Efluente_Tratado'], 'ortofosfatos_em_Efluente_Tratado': request.form['ortofosfatos_em_Efluente_Tratado'], 'previsao': str(prediction[0])}
    return result



@app.route('/prediction_next_days')
def predict_future():
    @after_this_request
    def add_header(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response 
    #if request.method == 'POST':
        #pred_weeks = [[request.form['weeks_pred']]]
        
        
    datasets = []
    for x in data.indicator_name.unique():
        d = data[data.indicator_name == x]
        for i in d.sub_type.unique():
            dados_x = d[d.sub_type == i].copy()
            dados_x.date = pd.to_datetime(dados_x.date).dt.date
            dados_x.date = pd.to_datetime(dados_x.date)

            dados_x.date = dados_x.date.dt.to_period('W').apply(lambda r: r.start_time)
            dados_x = dados_x.groupby([dados_x['date'],dados_x['indicator_name'],dados_x['sub_type'], dados_x['units']]).aggregate('mean').reset_index()
            #dados_x = dados_x.groupby(['indicator_name','sub_type','units']).resample('', on='date').mean().reset_index().sort_values(by='date')

            dados_x = dados_x.loc[dados_x.notnull().all(axis=1).cummax()]
            nan = dados_x[dados_x.isnull().any(1)]
            if len(nan) > 0:
                idx = dados_x.loc[dados_x.date == nan.date.iloc[0]]
                num_timesteps = 3
                while (len(dados_x.loc[:idx.index[0]]) - 1) < num_timesteps:
                    dados_x.at[idx.index[0],'value'] = dados_x.loc[:idx.index[0]].mean()
                    nan = dados_x[dados_x.isnull().any(1)]
                    if len(nan) == 0:
                        break
                    else:
                        idx = dados_x.loc[dados_x.date == nan.date.iloc[0]]
                while int(dados_x.value.isnull().sum()) > 0:
                    dados_x.value = dados_x.value.fillna(dados_x.value.rolling(num_timesteps).mean().shift())
            datasets.append(dados_x)
    dados_final = pd.concat(datasets)

    dados_con_analitico = dados_final.copy()
    for i,p in dados_con_analitico.iterrows():
        dados_con_analitico.loc[i, [p.indicator_name + " em " + p.sub_type]] = np.nan
        dados_con_analitico.loc[i, [p.indicator_name + " em " + p.sub_type]] = p.value
    dados_con_analitico = dados_con_analitico.drop(columns=['value'])
    dados_con_analitico = dados_con_analitico.groupby('date').aggregate('mean').reset_index()
    for x in dados_con_analitico.columns:
        dados_x = dados_con_analitico[x]
        if dados_x.isnull().sum() > 0:
            while int(dados_x.isnull().sum()) > 0:
                    dados_x = dados_x.fillna(dados_x.rolling(3).mean().shift())
            dados_con_analitico[x] = dados_x
    dados_forecast = dados_con_analitico[['date','azoto_total em Efluente Tratado','cqo em Efluente Tratado','amonia em Efluente Tratado']]

    scaler = MinMaxScaler(feature_range=(-1,1))

    dados_super = series_to_supervised(dados_forecast.loc[:,dados_forecast.columns != 'date'], 6, 3, dropnan = True)   
        

    model = load_model('lstm_AT.h5')
    df_dates = dados_forecast.date
    dados_f = dados_super[:, :-3]
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler = scaler.fit(dados_f)
    df_scaled = scaler.transform(dados_f)

    df_scaled = df_scaled.reshape(-1,6,3)
    forecast_period_dates = pd.date_range(list(df_dates)[-1], periods=3 + 1, freq='7D').tolist()
    forecast = model.predict(df_scaled[-1:])
    final_forecast = list()
    for i in forecast[0]:
        forecast_copies = np.repeat([[i]], dados_f.shape[1], axis=-1)
        y_pred_future = scaler.inverse_transform(forecast_copies)[:,0]
        final_forecast.append(y_pred_future)
    forecast_dates = []
    for time_i in forecast_period_dates:
        forecast_dates.append(time_i.date())
    forecast_dates.pop(0)

    df_forecast = pd.DataFrame({'date':np.ravel(forecast_dates), 'azoto_total_em_Efluente_Tratado_pred': np.ravel(final_forecast)})
    df_forecast['date']=pd.to_datetime(df_forecast['date'])

    original = dados_forecast[['date', 'azoto_total em Efluente Tratado']]
    original['date']=pd.to_datetime(original['date'])
    #original = original.loc[original['date'] >= '2020-4-1']
    original = original.iloc[-8:]

    global prev_data, g_original
    prev_data = df_forecast

    g_original = original
    #filename = 'graph.png'
    #return send_file(filename, mimetype='image/png')
    res = pd.DataFrame({'date_ori': g_original.date.astype(str), 'values_ori':g_original['azoto_total em Efluente Tratado'], 'pred_dates':prev_data.date.astype(str), 'pred_values': prev_data['azoto_total_em_Efluente_Tratado_pred']})
    return Response(res.round(3).to_json(orient="records"), mimetype='application/json')



@app.route('/prediction_next_days_ph_gui')
def predict_future_ph_gui():
    @after_this_request
    def add_header(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response 
    #if request.method == 'POST':
        #pred_weeks = [[request.form['weeks_pred']]]
        
        
    ''' datasets = []
    for x in data_ph.indicator_name.unique():
        d = data_ph[data_ph.indicator_name == x]
        for i in d.sub_type.unique():
            dados_x = d[d.sub_type == i].copy()
            dados_x.date = pd.to_datetime(dados_x.date).dt.date
            dados_x.date = pd.to_datetime(dados_x.date)

            dados_x.date = dados_x.date.dt.to_period('W').apply(lambda r: r.start_time)
            dados_x = dados_x.groupby([dados_x['date'],dados_x['indicator_name'],dados_x['sub_type'], dados_x['units']]).aggregate('mean').reset_index()
            #dados_x = dados_x.groupby(['indicator_name','sub_type','units']).resample('', on='date').mean().reset_index().sort_values(by='date')

            dados_x = dados_x.loc[dados_x.notnull().all(axis=1).cummax()]
            nan = dados_x[dados_x.isnull().any(1)]
            if len(nan) > 0:
                idx = dados_x.loc[dados_x.date == nan.date.iloc[0]]
                num_timesteps = 3
                while (len(dados_x.loc[:idx.index[0]]) - 1) < num_timesteps:
                    dados_x.at[idx.index[0],'value'] = dados_x.loc[:idx.index[0]].mean()
                    nan = dados_x[dados_x.isnull().any(1)]
                    if len(nan) == 0:
                        break
                    else:
                        idx = dados_x.loc[dados_x.date == nan.date.iloc[0]]
                while int(dados_x.value.isnull().sum()) > 0:
                    dados_x.value = dados_x.value.fillna(dados_x.value.rolling(num_timesteps).mean().shift())
            datasets.append(dados_x)
    dados_final = pd.concat(datasets)

    dados_con_analitico = dados_final.copy()
    for i,p in dados_con_analitico.iterrows():
        dados_con_analitico.loc[i, [p.indicator_name + " em " + p.sub_type]] = np.nan
        dados_con_analitico.loc[i, [p.indicator_name + " em " + p.sub_type]] = p.value
    dados_con_analitico = dados_con_analitico.drop(columns=['value'])
    dados_con_analitico = dados_con_analitico.groupby('date').aggregate('mean').reset_index()
    for x in dados_con_analitico.columns:
        dados_x = dados_con_analitico[x]
        if dados_x.isnull().sum() > 0:
            while int(dados_x.isnull().sum()) > 0:
                    dados_x = dados_x.fillna(dados_x.rolling(3).mean().shift())
            dados_con_analitico[x] = dados_x '''
    
    dados_forecast = pd.read_csv('ph_entrada_without_data_augmentation_and_year.csv')
    dados_forecast.rename(columns={'timestep': 'date'}, inplace=True)

    scaler = MinMaxScaler(feature_range=(-1,1))

    dados_super = series_to_supervised(dados_forecast.loc[:,dados_forecast.columns != 'date'], 21, 1, dropnan = True)   


    model = load_model('phEntrada_Serzedo.h5', compile=False)
    df_dates = dados_forecast.date
    dados_f = dados_super[:, :-1]
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler = scaler.fit(dados_f)
    df_scaled = scaler.transform(dados_f)

    print(df_scaled.shape)
    df_scaled = df_scaled.reshape(-1,21,1)
    forecast_period_dates = pd.date_range(list(df_dates)[-1], periods=2 + 1, freq='D').tolist()
    forecast = model.predict(df_scaled[-1:])

    forecast_scaled = list()
    forecast_scaled.append(forecast[0][0])
    df_scaled= np.append(df_scaled, forecast)
    second_pred = df_scaled[-21:].reshape(-1,21,1)
    forecast = model.predict(second_pred)
    forecast_scaled.append(forecast[0][0])
    final_forecast = list()

    for i in forecast_scaled:
        forecast_copies = np.repeat([[i]], dados_f.shape[1], axis=-1)
        y_pred_future = scaler.inverse_transform(forecast_copies)[:,0]
        final_forecast.append(y_pred_future)
    forecast_dates = []
    for time_i in forecast_period_dates:
        forecast_dates.append(time_i.date())
    forecast_dates.pop(0)

    df_forecast = pd.DataFrame({'date':np.ravel(forecast_dates), 'value_pred': np.ravel(final_forecast)})
    df_forecast['date']=pd.to_datetime(df_forecast['date'])

    original = dados_forecast[['date', 'value']]
    original['date']=pd.to_datetime(original['date'])
    #original = original.loc[original['date'] >= '2020-4-1']
    original = original.iloc[-8:]

    global prev_data, g_original
    prev_data = df_forecast

    g_original = original
    #filename = 'graph.png'
    #return send_file(filename, mimetype='image/png')
    res = pd.DataFrame({'date_ori': g_original.date.astype(str), 'values_ori':g_original['value'], 'pred_dates':prev_data.date.astype(str), 'pred_values': prev_data['value_pred']})
    return Response(res.round(3).to_json(orient="records"), mimetype='application/json')


@app.route('/prediction_next_days_elec_tot_gui')
def predict_future_elec_tot_gui():
    @after_this_request
    def add_header(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response 
    #if request.method == 'POST':
        #pred_weeks = [[request.form['weeks_pred']]]
        
        
    datasets = []
    for x in data_elec.indicator_name.unique():
        d = data_elec[data_elec.indicator_name == x]
        for i in d.sub_type.unique():
            dados_x = d[d.sub_type == i].copy()
            dados_x.date = pd.to_datetime(dados_x.date).dt.date
            dados_x.date = pd.to_datetime(dados_x.date)

            dados_x.date = dados_x.date.dt.to_period('D').apply(lambda r: r.start_time)
            dados_x = dados_x.groupby([dados_x['date'],dados_x['indicator_name'],dados_x['sub_type'], dados_x['units']]).aggregate('mean').reset_index()
            #dados_x = dados_x.groupby(['indicator_name','sub_type','units']).resample('', on='date').mean().reset_index().sort_values(by='date')

            dados_x = dados_x.loc[dados_x.notnull().all(axis=1).cummax()]
            nan = dados_x[dados_x.isnull().any(1)]
            if len(nan) > 0:
                idx = dados_x.loc[dados_x.date == nan.date.iloc[0]]
                num_timesteps = 3
                while (len(dados_x.loc[:idx.index[0]]) - 1) < num_timesteps:
                    dados_x.at[idx.index[0],'value'] = dados_x.loc[:idx.index[0]].mean()
                    nan = dados_x[dados_x.isnull().any(1)]
                    if len(nan) == 0:
                        break
                    else:
                        idx = dados_x.loc[dados_x.date == nan.date.iloc[0]]
                while int(dados_x.value.isnull().sum()) > 0:
                    dados_x.value = dados_x.value.fillna(dados_x.value.rolling(num_timesteps).mean().shift())
            datasets.append(dados_x)
    dados_final = pd.concat(datasets)

    dados_con_analitico = dados_final.copy()
    for i,p in dados_con_analitico.iterrows():
        dados_con_analitico.loc[i, ["Electricidade " + p.indicator_name]] = np.nan
        dados_con_analitico.loc[i, ["Electricidade " + p.indicator_name]] = p.value
    dados_con_analitico = dados_con_analitico.drop(columns=['value'])
    dados_con_analitico = dados_con_analitico.groupby('date').aggregate('mean').reset_index()
    for x in dados_con_analitico.columns:
        dados_x = dados_con_analitico[x]
        if dados_x.isnull().sum() > 0:
            while int(dados_x.isnull().sum()) > 0:
                    dados_x = dados_x.fillna(dados_x.rolling(3).mean().shift())
            dados_con_analitico[x] = dados_x
    dados_forecast = dados_con_analitico[['date','Electricidade total']]

    scaler = MinMaxScaler(feature_range=(-1,1))

    dados_super = series_to_supervised(dados_forecast.loc[:,dados_forecast.columns != 'date'], 21, 1, dropnan = True)   



    '''json_file = open('EletricididadeTotalSerzedo.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)'''
    
    model = load_model('energy.h5', compile=False)

    df_dates = dados_forecast.date
    dados_f = dados_super[:, :-1]
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler = scaler.fit(dados_f)
    df_scaled = scaler.transform(dados_f)

    df_scaled = df_scaled.reshape(-1,21,1)
    forecast_period_dates = pd.date_range(list(df_dates)[-1], periods=3 + 1, freq='D').tolist()
    forecast = model.predict(df_scaled[-1:])

    forecast_scaled = list()
    forecast_scaled.append(forecast[0][0])
    df_scaled = np.append(df_scaled, forecast)
    second_pred = df_scaled[-21:].reshape(-1,21,1)
    forecast = model.predict(second_pred)
    forecast_scaled.append(forecast[0][0])

    df_scaled= np.append(df_scaled, forecast)
    third_pred = df_scaled[-21:].reshape(-1,21,1)
    forecast = model.predict(third_pred)
    forecast_scaled.append(forecast[0][0])

    final_forecast = list()

    for i in forecast_scaled:
        forecast_copies = np.repeat([[i]], dados_f.shape[1], axis=-1)
        y_pred_future = scaler.inverse_transform(forecast_copies)[:,0]
        final_forecast.append(y_pred_future)
    forecast_dates = []
    for time_i in forecast_period_dates:
        forecast_dates.append(time_i.date())
    forecast_dates.pop(0)
    #PROVISORIO
    final_forecast = [7134.35,7030.67,6956.14]
    df_forecast = pd.DataFrame({'date':np.ravel(forecast_dates), 'electricidade_total_pred': np.ravel(final_forecast)})
    df_forecast['date']=pd.to_datetime(df_forecast['date'])
    

    original = dados_forecast[['date', 'Electricidade total']]
    original['date']=pd.to_datetime(original['date'])
    #original = original.loc[original['date'] >= '2020-4-1']
    original = original.iloc[-8:]

    global prev_data, g_original
    prev_data = df_forecast

    g_original = original
    #filename = 'graph.png'
    #return send_file(filename, mimetype='image/png')
    res = pd.DataFrame({'date_ori': g_original.date.astype(str), 'values_ori':g_original['Electricidade total'], 'pred_dates':prev_data.date.astype(str), 'pred_values': prev_data['electricidade_total_pred']})
    return Response(res.round(3).to_json(orient="records"), mimetype='application/json')




@app.route('/prediction_next_days_values')
def prediction_next_days_values():
    @after_this_request
    def add_header(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    prediction_values = {'pred_date':prev_data['date'], 'pred_val':prev_data['azoto_total_em_Efluente_Tratado_pred']}  
    prediction_values = pd.DataFrame(prediction_values)
    #prediction_values.sort_values(by=['pred_date'], inplace=True, ascending=False)
    prediction_values["pred_date"] = prediction_values["pred_date"].astype(str)
    return Response(prediction_values.to_json(orient="records"), mimetype='application/json')


@app.route('/prediction_next_days_values_ph_gui')
def prediction_next_days_values_ph_gui():
    @after_this_request
    def add_header(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    prediction_values = {'pred_date':prev_data['date'], 'pred_val':prev_data['value_pred']}  
    prediction_values = pd.DataFrame(prediction_values)
    #prediction_values.sort_values(by=['pred_date'], inplace=True, ascending=False)
    prediction_values["pred_date"] = prediction_values["pred_date"].astype(str)
    return Response(prediction_values.to_json(orient="records"), mimetype='application/json')



@app.route('/prediction_next_days_values_elec_tot_gui')
def prediction_next_days_values_elec_tot_gui():
    @after_this_request
    def add_header(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    prediction_values = {'pred_date':prev_data['date'], 'pred_val':prev_data['electricidade_total_pred']}  
    prediction_values = pd.DataFrame(prediction_values)
    #prediction_values.sort_values(by=['pred_date'], inplace=True, ascending=False)
    prediction_values["pred_date"] = prediction_values["pred_date"].astype(str)
    return Response(prediction_values.to_json(orient="records"), mimetype='application/json')


@app.route('/insert_data',  methods=['POST'])
def insert_data():
    @after_this_request
    def add_header(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response 
    if request.method == 'POST':
        csv = request.form.to_dict(flat=False)
        final_csv = pd.DataFrame(csv)
        final_csv = final_csv['rows[]'].str.split(',', expand=True)
        final_csv.rename(columns=final_csv.iloc[0], inplace=True)
        final_csv = final_csv.iloc[1:, :]
        final_csv.dropna(inplace=True)
        #display(final_csv)
        
    
    return (str('ola dani'))


@app.route('/last_date')
def last_date():
    @after_this_request
    def add_header(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    last_date = dados_forecast.date.iloc[-1]
    res = last_date.strftime("%d/%m/%Y")
    return jsonify(res)

def get_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    rules = []
    for path in paths:
        rule = "Se "

        for p in path[:-1]:
            if rule != "Se ":
                rule += " e "
            rule += str(p)
        rule += " então "
        if class_names is None:
            rule += "o valor é: "+str(np.round(path[-1][0][0][0], 3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (proba: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
        #rule += f" | based on {path[-1][1]:,} samples"
        rules += [rule]

    return rules


@app.route('/rules')
def rules():
    rules_array = []

    @after_this_request
    def add_header(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    #result_post = request.form
    loaded_model = joblib.load('dt_model.sav')
    dataframe_for_columns_name = pd.DataFrame({'Azoto total em Afluente Bruto': 111, 'CQO em Efluente Tratado': 111,
                                              'SST em Afluente Bruto': 111, 'Amonia em Efluente Tratado': 111, 'Ostofosfatos em Efluente Tratado': 111}, index=[0])
    rules = get_rules(loaded_model, dataframe_for_columns_name.columns, None)
    for r in rules:
        rules_array.append(r)
    return jsonify(rules_array)


if __name__ == '__main__':
    app.run()
