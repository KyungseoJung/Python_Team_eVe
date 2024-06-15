# // T-Map appkey = 5RB8KXlDuB6uLZGhugCqS9OJMViZ73P93dRPbphu 


'''
# //#26 관리자(Administrator) 웹화면 디자인 구성 
# //#27 관리자 웹페이지 디자인 
# //#28 수리모형 코드 통합 - [경로 계산하기] 버튼 누르면, 수리모형 코드 실행되도록 - csv 파일 지정한 파일 위치에 저장되도록
# //#29 배터리 강화학습 코드 통합


# //#38 RabbitMQ를 통해 고객 화면으로부터 수신한 주문 정보를 txt 파일에 추가
# (서비스 요청 시간에 따라 각각 다른 파일에 저장)
# - 서비스 요청 시간이 0~420인 경우: E_01.txt 파일
# - 서비스 요청 시간이 480~900인 경우: E_02.txt 파일
# - 서비스 요청 시간이 960~1380인 경우: E_03.txt 파일

# //#43 현재 시간대의 배달기사 경로 표시
# //#46 고객이 입력한 차량정보와 상세주소가 RabbitMQ를 통해 관리자 화면으로 수신되고, static 폴더 안의 txt파일에 저장되도록

# //#0603 //#0604 강화학슴 관련 코드 추가
'''

'''
#60
cmd 창에 ipconfig를 통해 IPv4 주소를 찾아

1) -> 관리자 페이지는 해당 안 됨
모바일로 실행 시,
screen_Client_home.html의 url 주소는 IPv4 주소를 작성하도록
-> 매일 수정되니까 실행할 때마다 확인 - 수정 - 실행
-> app_Client.py 파일의 port 번호와 screen_Client_home.html 의 port번호를 동일하게 맞추기

2) RabbitMQ 주소 맞추기
var ws = new WebSocket("ws://127.0.0.1:15674/ws");  -> 로컬로 연결되어 있는 것
IPv4 주소를 찾아서 아래와 같이 수정해주기
예를 들어, 192.168.50.178이라면, var ws = new WebSocket("ws://192.168.50.178:15674/ws");
'''


# from flask import Flask, render_template
# from flask import jsonify, request # //#28 수리모형 코드 통합을 위한 import

import pandas as pd
import eve_0605_sys # //#28 수리모형 코드 통합 (Import 수리모형 함수를 포함한 Python file )

import pandas as pd
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import dash_daq as daq

app = Flask(__name__)

# //#0603 추가 페키지 업데이트
from dash import dash_table
import plotly.express as px
import plotly.graph_objs as go
import os
import glob
import subprocess

import json
import random

#===================================================================================================
# //#38 RabbitMQ를 통해 고객 화면으로부터 수신한 주문 정보를 txt 파일에 추가 - 여기부터

@app.route('/append_data', methods=['POST'])
def append_data():
    data = request.get_json()

    try:
        print("//#46 Received data:", data)
        xcoord = float(data['xcoord'])
        ycoord = float(data['ycoord'])
        demand = int(data['demand'])
        ready_time = int(data['ready_time'])
        due_date = int(data['due_date'])

# //#46 고객이 입력한 차량정보와 상세주소가 RabbitMQ를 통해 관리자 화면으로 수신되고, static 폴더 안의 txt파일에 저장되도록
        car_num = str(data['car_num'])
        detail_address = str(data['detail_address'])

        print("//#46 Parsed data:", xcoord, ycoord, demand, ready_time, due_date, car_num, detail_address)

        # 신청한 시간대별로 파일 위치 지정
        # - 서비스 요청 시간이 0~420인 경우: E_01.txt 파일
        # - 서비스 요청 시간이 480~900인 경우: E_02.txt 파일
        # - 서비스 요청 시간이 960~1380인 경우: E_03.txt 파일
        if 0 <= ready_time < 480:
            file_name = 'static/orderData/E_01.txt'
        elif 480 <= ready_time < 960:
            file_name = 'static/orderData/E_02.txt'
        elif 960 <= ready_time <= 1380:
            file_name = 'static/orderData/E_03.txt'
        
# //#46 고객이 입력한 차량정보와 상세주소가 RabbitMQ를 통해 관리자 화면으로 수신되고, static 폴더 안의 txt파일에 저장되도록
        # append_customer_to_file(file_name, xcoord, ycoord, demand, ready_time, due_date)
        append_customer_to_file(file_name, xcoord, ycoord, demand, ready_time, due_date, car_num, detail_address)
        
        print("//#46 Data appended to file:", file_name)


        return jsonify({"status": "success"}), 200
    except ValueError as e:
        print("//#46 Error:", str(e))

        return jsonify({"status": "error", "message": "Invalid data types"}), 400

# //#46 고객이 입력한 차량정보와 상세주소가 RabbitMQ를 통해 관리자 화면으로 수신되고, static 폴더 안의 txt파일에 저장되도록
# def append_customer_to_file(file_name, xcoord, ycoord, demand, ready_time, due_date):
def append_customer_to_file(file_name, xcoord, ycoord, demand, ready_time, due_date, car_num, detail_address):
    # with open(file_name, 'r') as file:    // #46
    with open(file_name, 'r',  encoding='utf-8') as file:
        lines = file.readlines()
        customer_lines = [line for line in lines if line.strip() and line.split()[0].isdigit()]
        last_customer_line = customer_lines[-1]
        last_cust_no = int(last_customer_line.split()[0])

    new_cust_no = last_cust_no + 1

    # //#46 고객이 입력한 차량정보와 상세주소가 RabbitMQ를 통해 관리자 화면으로 수신되고, static 폴더 안의 txt파일에 저장되도록
    # new_customer_info = f"{new_cust_no:<6} {xcoord:<20} {ycoord:<20} {demand:<8} {ready_time:<10} {due_date:<10} 0\n"
    new_customer_info = f"{new_cust_no:<6} {xcoord:<20} {ycoord:<20} {demand:<8} {ready_time:<10} {due_date:<10} {car_num:<10} {detail_address}\n"
    # new_customer_info = f"{new_cust_no:<6} {xcoord:<20} {ycoord:<20} {demand:<8} {ready_time:<10} {due_date:<10} {car_num:<20} {detail_address:<20} 0\n".encode('cp949', errors='ignore')

    

    with open(file_name, 'a', encoding='utf-8') as file:

        # new_customer_info = new_customer_info.encode('cp949', errors='ignore').decode('cp949')  #//#46 UTF-8 데이터를 CP949로 변환

        file.write(new_customer_info)
    print(f"//#46 Data written to file {file_name}: {xcoord},{ycoord},{demand},{ready_time},{due_date},{car_num},{detail_address}")


# //#38 RabbitMQ를 통해 고객 화면으로부터 수신한 주문 정보를 txt 파일에 추가 - 여기까지
#===================================================================================================

#===================================================================================================
# //#29 배터리 강화학습 코드 통합 - 여기부터

# Define Dash layout # //#0604 09:00 대쉬보드 업데이트 - 기존 강화학습 대쉬보드 삭제
dash_app = Dash(__name__, server=app, url_base_pathname='/dash/')
# Load initial data
battery_dfs = pd.read_csv('./data/batt1.csv')
battery_dfs['Timestamp'] = pd.to_datetime(battery_dfs['Timestamp'])

RL_data = pd.read_csv('./RL_data/batt1.csv')
RL_data['Timestamp'] = pd.to_datetime(RL_data['Timestamp'])

# Load customer data from text files
directory_path = './data_total/'
total_data = glob.glob(os.path.join(directory_path, '*.csv'))
total_list = []
label_txt = ['오전','오후','심야']
a = 0
for i in total_data:
    data_df = pd.read_csv(i)
    data_df['label'] = label_txt[a]
    a += 1
    total_list.append(data_df)
total_df = pd.concat(total_list)

# Group data by label
gain_by_label = total_df.groupby('label')['Total_Gain'].sum().reset_index()
distance_by_label = total_df.groupby('label')['Total_distance'].sum().reset_index()

# Calculating total sums
total_gain_sum = total_df['Total_Gain'].sum()
total_dista_sum = total_df['Total_distance'].sum()

# Creating donut charts for Total_Gain by label
fig_total_gain = go.Figure(go.Pie(
    labels=gain_by_label['label'],
    values=gain_by_label['Total_Gain'],
    hole=0.5,
    textinfo='label+percent',
    hoverinfo='label+value'
))
fig_total_gain.add_annotation(
    x=0.5, y=0.5,
    text=f'Total: {total_gain_sum} 원',
    showarrow=False,
    font=dict(size=15)
)
fig_total_gain.update_layout(
    title='시간대별 서비스 매출',
    showlegend=True,
    height=500,  # Increase the size of the graph
    width=500
)

# Creating donut charts for Total_distance by label
fig_total_dista = go.Figure(go.Pie(
    labels=distance_by_label['label'],
    values=distance_by_label['Total_distance'],
    hole=0.5,
    textinfo='label+percent',
    hoverinfo='label+value'
))
fig_total_dista.add_annotation(
    x=0.5, y=0.5,
    text=f'Total: {total_dista_sum} m',
    showarrow=False,
    font=dict(size=15)
)
fig_total_dista.update_layout(
    title='시간대별 이동 거리',
    showlegend=True,
    height=500,  # Increase the size of the graph
    width=500
)

# Load customer data from text files
directory_path = '../RabbitMQ-Administrator/static/orderData/'
txt_files = glob.glob(os.path.join(directory_path, '*.txt'))

df_list = []
for datafile in txt_files:
    with open(datafile, 'r') as file:
        lines = file.read().strip().split('\n')

    customer_data = lines[10:]
    data = {
        'CUST NO.': [line.split()[0] for line in customer_data],
        'XCOORD.': [float(line.split()[1]) for line in customer_data],
        'YCOORD.': [float(line.split()[2]) for line in customer_data],
        'DEMAND': [int(line.split()[3]) for line in customer_data],
        'READY TIME': [int(line.split()[4]) for line in customer_data],
        'DUE DATE': [int(line.split()[5]) for line in customer_data]
    }
    df_list.append(pd.DataFrame(data))

# Concatenate all DataFrames in the list into a single DataFrame
df_all = pd.concat(df_list, ignore_index=True)
demand_counts = df_all['DEMAND'].value_counts().reset_index()
demand_counts.columns = ['DEMAND', 'count']

# Define a custom pastel blue color palette

# Create the donut chart figure
fig = go.Figure(data=[go.Pie(
    labels=demand_counts['DEMAND'],
    values=demand_counts['count'],
    hole=0.5,
    textinfo='label+percent',
    hoverinfo='label+value'
)])

# Create histogram figure with pastel blue tones
histogram_fig = go.Figure()
for demand in df_all['DEMAND'].unique():
    df_demand = df_all[df_all['DEMAND'] == demand]
    histogram_fig.add_trace(go.Histogram(
        x=df_demand['READY TIME'] // 60,  # Bin by hour
        name=f'DEMAND {demand}',
        xbins=dict(start=0, end=24, size=1),  # 0 to 23 for hourly bins
        opacity=0.75,
    ))

histogram_fig.update_layout(
    title='요금제별 이용 현황 (20/25/30/35)',
    barmode='stack',  # Change to 'stack' to stack the bars
    xaxis_title='READY TIME',
    yaxis_title='Count of Customers',
    bargap=0.2,
    xaxis=dict(
        tickmode='linear',
        tick0=0,
        dtick=1,
        tickvals=list(range(24)),  # Ensure the x-axis goes from 0 to 23
        ticktext=[str(i) for i in range(24)]
    )
)


# Group data by 60-minute intervals and sum the DEMAND
df_all['READY TIME BINNED'] = (df_all['READY TIME'] // 60)
demand_sum_by_ready_time = df_all.groupby('READY TIME BINNED')['DEMAND'].sum().reset_index()

# Create the bar chart figure
bar_chart_fig = px.bar(
    demand_sum_by_ready_time,
    x='READY TIME BINNED',
    y='DEMAND',
    labels={'READY TIME BINNED': 'READY TIME (hours)', 'DEMAND': 'Sum of DEMAND'},
    title='시간대별 전력 제공량 Kwh',
    category_orders={'READY TIME BINNED': list(range(24))}  # Ensure the x-axis goes from 0 to 23
)

nav_bar = html.Nav([
    html.Ul([
        html.Li(html.A('뒤로가기', href='/', style={'text-decoration': 'none', 'color': '#000000', 'background': '#ffffff', 'padding': '10px 40px', 'cursor': 'pointer', 'padding': '10px 20px', 'border-radius': '5px'})),
        html.Li(html.A('배터리 관리', href='/dash/home', style={'text-decoration': 'none', 'color': '#000000', 'background': '#ffffff', 'padding': '10px 40px', 'cursor': 'pointer', 'padding': '10px 20px', 'border-radius': '5px'})),
        html.Li(html.A('배달 분석', href='/dash/', style={'text-decoration': 'none', 'color': '#000000', 'background': '#ffffff', 'padding': '10px 40px', 'cursor': 'pointer', 'padding': '10px 20px', 'border-radius': '5px'})),
    ], style={'list-style-type': 'none', 'margin': 0, 'padding': 0, 'display': 'flex', 'justify-content': 'flex-start'})
], style={'background-color': '#ffffff', 'padding': '10px 40px','text-align':'left' ,'box-shadow': '0 8px 8px rgba(0,0,0,0.10)'})

home_layout = html.Div([
    nav_bar,
    html.H1("eVe Battery Management System", style={'text-align': 'center', 'padding': '50px', 'color': '#2C3E50'}),
    html.Div([
      html.Div([
            
            html.Div([
                dcc.Graph(id='donut-chart-total-gain', figure=fig_total_gain, style={'background-color': '#ffffff',"height":"80vh"}),
                    html.Div([html.P([html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),
                                  f"심야 시간에 {gain_by_label['Total_Gain'][0]} 원",html.Br(),
                                  f"오전 시간에 {gain_by_label['Total_Gain'][1]} 원",html.Br(),
                                  f"오후 시간에 {gain_by_label['Total_Gain'][2]} 원",html.Br(),
                                  f"총 {total_gain_sum}원의 매출이 발생했습니다",html.Br(),
                                  ])],style={"background-color":"#ffffff"}),
            ],style={'display': 'grid','grid-template-columns': '1fr 1fr',"width":"100%", 'align-items': 'center',"background-color":"#ffffff"}),
            html.Div([
                dcc.Graph(id='donut-chart-total-dista', figure=fig_total_dista, style={'background-color': '#ffffff',"height":"80vh"}),
                    html.Div([html.P([html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),
                                  f"심야 시간에 {float(distance_by_label['Total_distance'][0]):.02f} m",html.Br(),
                                  f"오전 시간에 {float(distance_by_label['Total_distance'][1]):.02f} m",html.Br(),
                                  f"오후 시간에 {float(distance_by_label['Total_distance'][2]):.02f} m",html.Br(),
                                  f"총 {total_dista_sum}m 를 이동했습니다",html.Br(),
                                  ])],style={"background-color":"#ffffff"}),
            ],style={'display': 'grid','grid-template-columns': '1fr 1fr',"width":"100%", 'align-items': 'center',"background-color":"#ffffff"}),
        ], style={'display': 'grid', 'grid-template-columns': '1fr 1fr', 'align-items': 'stretch', 'background-color': '#F5F5F5', 'padding': '20px'}),

        html.Div([
            dcc.Graph(id='donut-chart', figure=fig, style={'background-color': '#F5F5F5'}),
            dcc.Graph(id='histogram', figure=histogram_fig, style={'background-color': '#F5F5F5'})
        ], style={'display': 'grid', 'grid-template-columns': '1fr 2fr', 'align-items': 'stretch', 'background-color': '#F5F5F5', 'padding': '20px'}),
        dcc.Graph(id='bar-chart', figure=bar_chart_fig, style={'background-color': '#F5F5F5', 'padding': '20px'})
    ], style={'background-color': '#ffffff'})
], style={'background-color': '#F5F5F5'})

# Define dashboard layout
dashboard_layout = html.Div([
    dcc.Store(id='selected-battery', data='batt1'),
    nav_bar,
    html.Div([
        html.Div([
            html.H1("배터리 관리 페이지", style={'color': '#000000'}),
            html.Div(
                id='Batt',
                children=[dcc.Dropdown(['batt1', 'batt2', 'batt3', 'batt4', 'batt5', 'batt6', 'batt7', 'batt8'], 'batt1', id='battery-dropdown', style={'color': 'black', 'font-size': 15, 'background-color': '#CEDBF6'})],
            ),
            html.P(
                [
                    "이 대시보드는 배터리 상태 모니터링 시스템입니다. ", html.Br(),
                    "실시간 데이터와 계획된 데이터를 비교하여 배터리의 충전 상태, 전류, 온도, 전압을 시각화합니다. ", html.Br(),
                    "배터리 선택에 따라 관련 데이터를 표시하며, 현재 시간과 현재 배터리 용량(%), 현재 배터리 온도를 제공합니다. ", html.Br(), html.Br(),
                ],
                style={'color': '#000000', 'border-bottom': '2.5px solid gray', 'padding': '0 10 0 10'}
            ),
            html.Div(
                id="control-panel-utc",
                children=[
                    daq.LEDDisplay(
                        id="control-panel-utc-component",
                        value="00:00",
                        label="Time",
                        size=40,
                        color="#0000CD",
                        backgroundColor="#ffffff",
                        style={'padding': '0 5 0 5', 'margin': '0 5 0 5'}
                    ),
                    dcc.Interval(id='interval-component-utc', interval=60 * 1000, n_intervals=0)
                ],
            ),
            html.Br(),
            html.Div([
                html.Div(
                    id="control-panel-elevation",
                    children=[
                        daq.Tank(
                            id="control-panel-elevation-component",
                            label={'label': 'Battery Capacity', 'style': {'color': 'black'}},
                            labelPosition='bottom',
                            min=0,
                            max=1,
                            showCurrentValue=True,
                            color="#303030",
                            height=250,
                        )
                    ],
                ),
                html.Div(
                    id="control-panel-temperature",
                    children=[
                        daq.Thermometer(
                            id="control-panel-temperature-component",
                            label="Temperature",
                            min=0,
                            max=400,
                            value=290,
                            units="Kelvin",
                            showCurrentValue=True,
                            color="#303030",
                            style={'color': '#F2F2F2'}
                        )
                    ],
                    style={'padding': 10, 'margin': 0, 'border-radius': '100px'}
                ),
                dcc.Interval(id='interval-component-now', interval=1 * 1000, n_intervals=0)
            ], style={'display': 'flex', 'justify-content': 'space-around', 'align-items': 'center', 'padding': 10, 'margin': 10}),
        ], style={'background-color': '#ffffff', 'margin':20, 'padding': '10px','border-radius': '8px'}),
        html.Div([
            html.Div(id="graph_1", children=[dcc.Graph(id='battery-graph_1'), dcc.Interval(id='interval-component_1', interval=1 * 1000, n_intervals=0)], style={'padding': 20, 'margin': 0, 'box-shadow': '0 2px 2px rgba(0,0,0,0.2)', 'background-color': '#F9FDFF','border-radius': '8px'}),
            html.Div(id="graph_2", children=[dcc.Graph(id='battery-graph_2'), dcc.Interval(id='interval-component_2', interval=1 * 1000, n_intervals=0)], style={'padding': 20, 'margin': 0, 'box-shadow': '0 2px 2px rgba(0,0,0,0.2)', 'background-color': '#F9FDFF','border-radius': '8px'}),
            html.Div(id="graph_3", children=[dcc.Graph(id='battery-graph_3'), dcc.Interval(id='interval-component_3', interval=1 * 1000, n_intervals=0)], style={'padding': 20, 'margin': 0, 'box-shadow': '0 2px 2px rgba(0,0,0,0.2)', 'background-color': '#F9FDFF','border-radius': '8px'}),
            html.Div(id="graph_4", children=[dcc.Graph(id='battery-graph_4'), dcc.Interval(id='interval-component_4', interval=1 * 1000, n_intervals=0)], style={'padding': 20, 'margin': 0, 'box-shadow': '0 2px 2px rgba(0,0,0,0.2)', 'background-color': '#F9FDFF','border-radius': '8px'}),
        ], style={'padding': 10, 'margin': 10, 'display': 'grid', 'gap': 20, 'grid-template-columns': 'repeat(2, 1fr)', 'grid-template-rows': 'repeat(2, 1fr)'})
        
    ], style={'display': 'grid', 'grid-template-columns': '1fr 2fr', 'align-items': 'stretch', 'background-color': '#F5F5F5'}),
], style={'color': '#000000', 'background-color': '#F5F5F5', 'padding': 1, 'margin': 1})

# Update layout based on URL
@dash_app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/dash/home':
        return dashboard_layout
    else:
        return home_layout

# Define Dash layout with URL routing
dash_app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Define Dash callbacks
@dash_app.callback(Output('control-panel-utc-component', 'value'), Input('interval-component-utc', 'n_intervals'))
def update_time(n):
    return datetime.now().strftime('%H:%M')

@dash_app.callback(Output('selected-battery', 'data'), Input('battery-dropdown', 'value'))
def update_selected_battery(selected_battery):
    return selected_battery

@dash_app.callback(
    [Output('control-panel-elevation-component', 'value'), Output('control-panel-temperature-component', 'value')],
    [Input('interval-component-now', 'n_intervals'), Input('selected-battery', 'data')]
)
def update_tank_values(n_intervals, selected_battery):
    RL_data = pd.read_csv(f'./RL_data/{selected_battery}.csv')
    RL_data['Timestamp'] = pd.to_datetime(RL_data['Timestamp'])
    current_time = datetime.now()
    data_now = RL_data[RL_data['Timestamp'] <= current_time].iloc[-1]
    return round(data_now['state'], 2), round(data_now['temp'], 2)

@dash_app.callback(
    [Output('battery-graph_1', 'figure'), Output('battery-graph_2', 'figure'), Output('battery-graph_3', 'figure'), Output('battery-graph_4', 'figure')],
    [Input('interval-component_1', 'n_intervals'), Input('interval-component_2', 'n_intervals'), Input('interval-component_3', 'n_intervals'), Input('interval-component_4', 'n_intervals'), Input('selected-battery', 'data')]
)
def update_graph_live(n1, n2, n3, n4, selected_battery):
    battery_dfs = pd.read_csv(f'./data/{selected_battery}.csv')
    battery_dfs['Timestamp'] = pd.to_datetime(battery_dfs['Timestamp'])
    RL_data = pd.read_csv(f'./RL_data/{selected_battery}.csv')
    RL_data['Timestamp'] = pd.to_datetime(RL_data['Timestamp'])

    current_time = datetime.now()
    real_time_data = RL_data[RL_data['Timestamp'] <= current_time]

    figures = []
    metrics = ['state', 'c_rate', 'temp', 'volt']
    titles = ['Battery SoC(%) Over Time', 'Battery C-rate Over Time', 'Battery Temperature(K) Over Time', 'Battery Voltage(V) Over Time']
    for metric, title in zip(metrics, titles):
        figures.append({
            'data': [
                {'x': battery_dfs['Timestamp'], 'y': battery_dfs[metric], 'type': 'line', 'name': 'CC', 'line': {'dash': 'dash', 'color': '#00FFFF'}, 'hoverinfo': 'y+name'},
                {'x': real_time_data['Timestamp'], 'y': real_time_data[metric], 'type': 'line', 'name': 'RL', 'line': {'color': '#0000CD'}, 'hoverinfo': 'y+name'},
            ],
            'layout': {
                'title': {
                    'text': title,
                    'x': 0.1,
                    'xanchor': 'left',
                    'font': {'size': 16, 'color': '#000000', 'bold': True}
                },
                 'transition': {'duration': 500, 'easing': 'cubic-in-out'},
            'uirevision': 'constant',
            'plot_bgcolor': '#F9FDFF',
            'paper_bgcolor': '#F9FDFF',
            'font': {'color': '#000000'},
            'legend': {'x': 0.99, 'y': 1.0, 'xanchor': 'left', 'yanchor': 'top'},
            }
        })
    return figures
# //#29 배터리 강화학습 코드 통합 - 여기까지
#===================================================================================================


@app.route('/calculate_path', methods=['POST'])
def calculate_path():
        # import eve_0521_test
    try:
        for i in range(1,4):

        # //#28 내가 불러올 데이터 - 버튼 하나로 3개 데이터 뱉도록
            datafile = f"C:/GitStudy/Python_Team_eVe/RabbitMQ-Administrator/static/orderData/E_0{i}.txt"

            # //#28 내가 지정하는 경로 (파일 저장)
            # //#28 fix: pickle_path 코드 수정 - 주석 처리
            # pickle_path = "C:/GitStudy/Python_Team_eVe/RabbitMQ-Administrator/static/all_k_shortest_paths.pickle_S_02"
            battery_csv_path = "C:/GitStudy/Python_Team_eVe/RabbitMQ-Administrator/static/"
            truck_csv_path = "C:/GitStudy/Python_Team_eVe/RabbitMQ-Administrator/static/"
            drawroute_json_path = "C:/GitStudy/Python_Team_eVe/RabbitMQ-Administrator/static/"   # //#43 현재 시간대의 배달기사 경로 표시 위해 필요한 json 파일

            # 추가
            solution_path =  "C:/GitStudy/Python_Team_eVe/RabbitMQ-Administrator/static/"
            # eve_0522_test3.solve(datafile, pickle_path, battery_csv_path, truck_csv_path) # //#28 fix: pickle_path 코드 수정 - 주석 처리 
            # //#43 현재 시간대의 배달기사 경로 표시 위해 필요한 json 파일 경로 추가ㅠ
            eve_0605_sys.solve(datafile, battery_csv_path, truck_csv_path, drawroute_json_path, solution_path)

            # Assuming there's a function in eve_0522_test.py to calculate the path and save a CSV
            # result = eve_0522_test.calculate_and_save()
            return jsonify({'success': True, 'message': '경로 계산 및 CSV 저장 완료!'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

    
@app.route('/')
def administrator_page():
    return render_template('screen_Administrator.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5004,debug=True)

    # host ="0.0.0.0"    - ipconfig를 통해 IPV4 주소로 접속 가능