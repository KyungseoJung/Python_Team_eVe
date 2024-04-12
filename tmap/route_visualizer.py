from flask import Flask, render_template, request
import requests
import random

app = Flask(__name__)

# Set up the Tmap API endpoint URL and headers
url = 'https://apis.openapi.sk.com/tmap/routes/sequential/3.0'
headers = {'Content-Type': 'application/json'}

# 실제 API 키로 대체
api_key = '5RB8KXlDuB6uLZGhugCqS9OJMViZ73P93dRPbphu'    # 'GDfiIP3kG14El52XlO9NCa2WhUHkej5w65RzM412'

# Define the home page with a form to input the start and end locations
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        start_lat = request.form['start_lat']
        start_lon = request.form['start_lon']
        end_lat = request.form['end_lat']
        end_lon = request.form['end_lon']

        # Generate 10 random waypoints in Gangnam-gu, Seoul
        waypoints = []
        for i in range(10):
            lat = str(random.uniform(37.495, 37.525))
            lon = str(random.uniform(127.020, 127.060))
            waypoints.append({
                'lon': lon,
                'lat': lat
            })

        # Set up the request parameters
        params = {
            'appKey': api_key,
            'startX': start_lon,
            'startY': start_lat,
            'endX': end_lon,
            'endY': end_lat,
            'reqCoordType': 'WGS84GEO',
            'resCoordType': 'WGS84GEO',
            'searchOption': 0,
            'waypoints': waypoints
        }

        # Send the API request
        response = requests.get(url, headers=headers, params=params)

        # Render the result page with the response data
        return render_template('result.html', data=response.json())

    # Render the home page with the form
    return render_template('home.html')

# Define the result page to display the route data
@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
