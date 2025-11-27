import requests
import pandas as pd
from io import StringIO

# 요청할 위치 및 파라미터 설정
latitude = 35.54
longitude = 126.85
start_date = "20250814"  # YYYYMMDD 형식
end_date   = "20251101"  # 원하는 종료일

# 가져올 파라미터 (예: 수평면 일사량)
parameters = "ALLSKY_SFC_SW_DWN"

# 커뮤니티 종류: 태양광/재생에너지 관련이면 “RE”
community = "RE"

# 요청 URL 구성
url = ("https://power.larc.nasa.gov/api/temporal/daily/point"f"?parameters={parameters}"f"&community={community}"
    f"&longitude={longitude}"
    f"&latitude={latitude}"
    f"&start={start_date}"
    f"&end={end_date}"
    "&format=CSV"
)

# 데이터 요청
resp = requests.get(url)
resp.raise_for_status()
csv_text = resp.text

# CSV → DataFrame 변환
df = pd.read_csv(StringIO(csv_text), skiprows=11)  # 헤더 정보가 좀 있음
print(df.head())
print(df.tail())
print("총 행 수:", len(df))
