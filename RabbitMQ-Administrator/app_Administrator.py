# // T-Map appkey = 5RB8KXlDuB6uLZGhugCqS9OJMViZ73P93dRPbphu 

# //#26 관리자(Administrator) 웹화면 디자인 구성 
# //#27 관리자 웹페이지 디자인 
# //#28 수리모형 코드 통합 - [경로 계산하기] 버튼 누르면, 수리모형 코드 실행되도록 - csv 파일 지정한 파일 위치에 저장되도록
# //#29 배터리 강화학습 코드 통합

# from flask import Flask, render_template
# from flask import jsonify, request # //#28 수리모형 코드 통합을 위한 import

import pandas as pd
import eve_0523_test1 # //#28 수리모형 코드 통합 (Import 수리모형 함수를 포함한 Python file )

import pandas as pd
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import dash_daq as daq

app = Flask(__name__)

#===================================================================================================
# //#29 배터리 강화학습 코드 통합 - 여기부터

# Initialize Dash app
dash_app = Dash(__name__, server=app, url_base_pathname='/dash/')

# Load initial data
battery_dfs = pd.read_csv('./data/batt1.csv')
battery_dfs['Timestamp'] = pd.to_datetime(battery_dfs['Timestamp'])

RL_data = pd.read_csv('./RL_data/batt1.csv')
RL_data['Timestamp'] = pd.to_datetime(RL_data['Timestamp'])

# Define Dash layout
dash_app.layout = html.Div([
    dcc.Store(id='selected-battery', data='batt1'),
    html.Nav([
        html.Ul([
            html.Li(html.A('Home', href='http://127.0.0.1:5004', style={'color': '#F2F2F2', 'text-decoration': 'none', 'font-size': '24px', 'background-color': '#4A4A4A', 'padding': '10px 20px', 'border-radius': '5px'})),
        ], style={'list-style-type': 'none', 'margin': 0, 'padding': 0, 'display': 'flex', 'justify-content': 'flex-start'})
    ], style={'background-color': '#1E1E1E', 'padding': '10px', 'margin': 0}),
    html.Div([
        html.Div([
            html.H1("About Battery", style={'color': '#F2F2F2'}),
            html.Div(
                id='Batt',
                children=[dcc.Dropdown(['batt1', 'batt2', 'batt3', 'batt4', 'batt5'], 'batt1', id='battery-dropdown', style={'color': 'black', 'font-size': 15, 'background-color': '#2B2B2B'})],
            ),
            html.P(
                [
                    "이 대시보드는 배터리 상태 모니터링 시스템입니다. ", html.Br(),
                    "실시간 데이터와 계획된 데이터를 비교하여 배터리의 충전 상태, 전류, 온도, 전압을 시각화합니다. ", html.Br(),
                    "배터리 선택에 따라 관련 데이터를 표시하며, 현재 시간을 포함한 다양한 대시보드 요소를 제공합니다. ", html.Br(), html.Br(),
                ],
                style={'color': '#F2F2F2', 'border-bottom': '2.5px solid gray', "padding": '0 10 0 10'}
            ),
            html.Div(
                id="control-panel-utc",
                children=[
                    daq.LEDDisplay(
                        id="control-panel-utc-component",
                        value="00:00",
                        label="Time",
                        size=40,
                        color="#fec036",
                        backgroundColor="#303030",
                        style={"padding": '0 5 0 5', "margin": '0 5 0 5'}
                    ),
                    dcc.Interval(
                        id='interval-component-utc',
                        interval=60 * 1000,  # Update every minute
                        n_intervals=0
                    )
                ],
            ),
            html.Div([html.Br()]),
            html.Div([
                html.Div(
                    id="control-panel-elevation",
                    children=[
                        daq.Tank(
                            id="control-panel-elevation-component",
                            label="Battery Capacity",
                            min=0,
                            max=1,
                            value=0.8,  # 초기 값 설정
                            units="%",
                            showCurrentValue=True,
                            color="#303030",
                            style={'color': '#F2F2F2'}  # 라벨의 색상 수정
                        )
                    ],
                    style={"padding": 0, "margin": 0}
                ),
                html.Div(
                    id="control-panel-temperature",
                    children=[
                        daq.Tank(
                            id="control-panel-temperature-component",
                            label="Temperature",
                            min=0,
                            max=400,
                            value=290,  # 초기 값 설정
                            units="Kelvin",
                            showCurrentValue=True,
                            color="#303030",
                            style={'color': '#F2F2F2'}
                        )
                    ],
                    style={"padding": 0, "margin": 0}
                ),
                dcc.Interval(
                    id='interval-component-now',
                    interval=1 * 1000,  # Update every second
                    n_intervals=0
                )
            ], style={"display": "flex", "justify-content": "space-around", "padding": 0, "margin": 0}),
        ], style={'background-color': '#1E1E1E', "margin": 0, "padding": '10px'}),
        html.Div([
            html.Div(
                id="graph_1",
                children=[
                    dcc.Graph(id='battery-graph_1'),
                    dcc.Interval(
                        id='interval-component_1',
                        interval=1 * 1000,
                        n_intervals=0
                    )
                ],
                style={"padding": 0, "margin": 0, "height": "100%"}
            ),
            html.Div(
                id="graph_2",
                children=[
                    dcc.Graph(id='battery-graph_2'),
                    dcc.Interval(
                        id='interval-component_2',
                        interval=60 * 1000,
                        n_intervals=0
                    )
                ],
                style={"padding": 0, "margin": 0, "height": "100%"}
            ),
            html.Div(
                id="graph_3",
                children=[
                    dcc.Graph(id='battery-graph_3'),
                    dcc.Interval(
                        id='interval-component_3',
                        interval=60 * 1000,
                        n_intervals=0
                    )
                ],
                style={"padding": 0, "margin": 0, "height": "100%"}
            ),
            html.Div(
                id="graph_4",
                children=[
                    dcc.Graph(id='battery-graph_4'),
                    dcc.Interval(
                        id='interval-component_4',
                        interval=60 * 1000,
                        n_intervals=0
                    )
                ],
                style={"padding": 0, "margin": 0, "height": "100%"}
            )
        ], style={
            'color': '#BFBFBF',
            "padding": 0,
            "margin": 0,
            'display': 'grid',
            'gap': 0,
            "grid-template-columns": 'repeat(2, 1fr)',
            "grid-template-rows": 'repeat(2, 1fr)'
        })
    ], style={'display': 'grid', "grid-template-columns": "1fr 2fr", 'align-items': 'stretch', 'background-color': '#ffffff'})
], style={'color': '#000000', 'background-color': '#ffffff', "padding": 0, "margin": 0})

# Define Dash callbacks
@dash_app.callback(
    Output('control-panel-utc-component', 'value'),
    Input('interval-component-utc', 'n_intervals')
)
def update_time(n):
    current_time = datetime.now()
    return current_time.strftime('%H:%M')

@dash_app.callback(
    Output('selected-battery', 'data'),
    Input('battery-dropdown', 'value')
)
def update_selected_battery(selected_battery):
    return selected_battery

@dash_app.callback(
    [
        Output('control-panel-elevation-component', 'value'),
        Output('control-panel-temperature-component', 'value'),
    ],
    [Input('interval-component-now', 'n_intervals'), Input('selected-battery', 'data')]
)
def update_tank_values(n_intervals, selected_battery):
    RL_data = pd.read_csv(f'./RL_data/{selected_battery}.csv')
    RL_data['Timestamp'] = pd.to_datetime(RL_data['Timestamp'])
    current_time = datetime.now()
    data_now = RL_data[RL_data['Timestamp'] <= current_time].iloc[-1]  # Get the latest data available
    state = round(data_now['state'], 2)
    temp = round(data_now['temp'], 2)
    return state, temp

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

    figure_1 = {
        'data': [
            {'x': battery_dfs['Timestamp'], 'y': battery_dfs['state'], 'type': 'line', 'name': 'CC', 'line': {'dash': 'dash', 'color': '#D09002'}, 'hoverinfo': 'y+name'},
            {'x': real_time_data['Timestamp'], 'y': real_time_data['state'], 'type': 'line', 'name': 'RL', 'line': {'color': '#fec036'}, 'hoverinfo': 'y+name'},
        ],
        'layout': {
            'title': 'Battery State of Charge Over Time',
            'transition': {'duration': 500, 'easing': 'cubic-in-out'},
            'yaxis': {'title': 'State of Charge (%)'},
            'uirevision': 'constant',
            'plot_bgcolor': '#2B2B2B',
            'paper_bgcolor': '#2B2B2B',
            'font': {'color': '#D5D5D5'},
            'legend': {'x': 0.99, 'y': 1.0, 'xanchor': 'left', 'yanchor': 'top'},
            'xaxis': {
                'gridcolor': '#7f7f7f',
                'gridwidth': 1,
            },
        }
    }

    figure_2 = {
        'data': [
            {'x': battery_dfs['Timestamp'], 'y': battery_dfs['current'], 'type': 'line', 'name': 'CC', 'line': {'dash': 'dash', 'color': '#D09002'}, 'hoverinfo': 'y+name'},
            {'x': real_time_data['Timestamp'], 'y': real_time_data['current'], 'type': 'line', 'name': 'RL', 'line': {'color': '#fec036'}, 'hoverinfo': 'y+name'},
        ],
        'layout': {
            'title': 'Battery Current Over Time',
            'transition': {'duration': 500, 'easing': 'cubic-in-out'},
            'yaxis': {'title': 'Current (A)'},
            'uirevision': 'constant',
            'plot_bgcolor': '#2B2B2B',
            'paper_bgcolor': '#2B2B2B',
            'font': {'color': '#D5D5D5'},
            'legend': {'x': 0.99, 'y': 1.0, 'xanchor': 'left', 'yanchor': 'top'},
            'xaxis': {
                'gridcolor': '#7f7f7f',
                'gridwidth': 1,
            },
        }
    }

    figure_3 = {
        'data': [
            {'x': battery_dfs['Timestamp'], 'y': battery_dfs['temp'], 'type': 'line', 'name': 'CC', 'line': {'dash': 'dash', 'color': '#D09002'}, 'hoverinfo': 'y+name'},
            {'x': real_time_data['Timestamp'], 'y': real_time_data['temp'], 'type': 'line', 'name': 'RL', 'line': {'color': '#fec036'}, 'hoverinfo': 'y+name'},
        ],
        'layout': {
            'title': 'Battery Temperature Over Time',
            'transition': {'duration': 500, 'easing': 'cubic-in-out'},
            'yaxis': {'title': 'Temperature (°C)'},
            'uirevision': 'constant',
            'plot_bgcolor': '#2B2B2B',
            'paper_bgcolor': '#2B2B2B',
            'font': {'color': '#D5D5D5'},
            'legend': {'x': 0.99, 'y': 1.0, 'xanchor': 'left', 'yanchor': 'top'},
            'xaxis': {
                'gridcolor': '#7f7f7f',
                'gridwidth': 1,
            },
        }
    }

    figure_4 = {
        'data': [
            {'x': battery_dfs['Timestamp'], 'y': battery_dfs['volt'], 'type': 'line', 'name': 'CC', 'line': {'dash': 'dash', 'color': '#D09002'}, 'hoverinfo': 'y+name'},
            {'x': real_time_data['Timestamp'], 'y': real_time_data['volt'], 'type': 'line', 'name': 'RL', 'line': {'color': '#fec036'}, 'hoverinfo': 'y+name'},
        ],
        'layout': {
            'title': 'Battery Voltage Over Time',
            'transition': {'duration': 500, 'easing': 'cubic-in-out'},
            'yaxis': {'title': 'Voltage (V)'},
            'uirevision': 'constant',
            'plot_bgcolor': '#2B2B2B',
            'paper_bgcolor': '#2B2B2B',
            'font': {'color': '#D5D5D5'},
            'legend': {'x': 0.99, 'y': 1.0, 'xanchor': 'left', 'yanchor': 'top'},
            'xaxis': {
                'gridcolor': '#7f7f7f',
                'gridwidth': 1,
            },
        }
    }

    return figure_1, figure_2, figure_3, figure_4

# //#29 배터리 강화학습 코드 통합 - 여기까지
#===================================================================================================


@app.route('/calculate_path', methods=['POST'])
def calculate_path():
        # import eve_0521_test
    try:
        # //#28 내가 불러올 데이터
        datafile = "C:/GitStudy/Python_Team_eVe/RabbitMQ-Administrator/orderData/S_03.txt"

        # //#28 내가 지정하는 경로 (파일 저장)
        # //#28 fix: pickle_path 코드 수정 - 주석 처리
        # pickle_path = "C:/GitStudy/Python_Team_eVe/RabbitMQ-Administrator/static/all_k_shortest_paths.pickle_S_02"
        battery_csv_path = "C:/GitStudy/Python_Team_eVe/RabbitMQ-Administrator/static/"
        truck_csv_path = "C:/GitStudy/Python_Team_eVe/RabbitMQ-Administrator/static/"

        # eve_0522_test3.solve(datafile, pickle_path, battery_csv_path, truck_csv_path) # //#28 fix: pickle_path 코드 수정 - 주석 처리 
        eve_0523_test1.solve(datafile, battery_csv_path, truck_csv_path)

        # Assuming there's a function in eve_0522_test.py to calculate the path and save a CSV
        # result = eve_0522_test.calculate_and_save()
        return jsonify({'success': True, 'message': '경로 계산 및 CSV 저장 완료!'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

    
@app.route('/')
def administrator_page():
    return render_template('screen_Administrator.html')


if __name__ == '__main__':
    app.run(port=5004,debug=True)