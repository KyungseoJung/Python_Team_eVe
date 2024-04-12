from threading import Thread
import requests

def draw_polyline_on_map(start_point, end_point):
    api_key = "5RB8KXlDuB6uLZGhugCqS9OJMViZ73P93dRPbphu"  # 여기에 T-map API 키를 입력하세요.
    url = f"https://api.tmap.co.kr/route/pedestrian?version=1&format=json&appKey={api_key}&startX={start_point.split(',')[1]}&startY={start_point.split(',')[0]}&endX={end_point.split(',')[1]}&endY={end_point.split(',')[0]}"
    
    def draw_polyline():
        try:
            response = requests.get(url, timeout=10)  # timeout 값 추가
            if response.status_code == 200:
                print("Polyline drawn successfully on T-map.")
            else:
                print("Failed to draw polyline on T-map.")
        except Exception as e:
            print("Error occurred while drawing polyline:", str(e))

    t = Thread(target=draw_polyline)
    t.start()
