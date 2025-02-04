import os
from dotenv import load_dotenv
import pyupbit
import pandas as pd
import json
from openai import OpenAI
import ta
from ta.utils import dropna
import time
import requests
import base64
from PIL import Image
import io
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, ElementClickInterceptedException, WebDriverException, NoSuchElementException
import logging
from youtube_transcript_api import YouTubeTranscriptApi
from pydantic import BaseModel
import sqlite3
from datetime import datetime, timedelta
import pytz
from tqdm import tqdm

# 한국 시간 (KST) 적용
KST = pytz.timezone("Asia/Seoul")

# .env 파일에 저장된 환경 변수를 불러오기 (API 키 등)
load_dotenv()

# 로깅 설정 - 로그 레벨을 INFO로 설정하여 중요 정보 출력
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("WDM").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("openai._base_client").setLevel(logging.ERROR)

if not logger.hasHandlers():  # 기존 핸들러가 없을 때만 설정 적용
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s: %(message)s",
        handlers=[logging.StreamHandler()]  # 콘솔 출력 핸들러 추가
    )

logger.propagate = False  # 중복 메시지 출력 방지

# 실행할 요일 및 시간 설정

EXECUTION_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]  # 실행할 요일
EXECUTION_TIMES = [
    "01:00", "05:00", "09:00", "10:00", "11:00", "12:00",
    "13:00", "14:00", "15:00", "16:00", "17:00", "18:00",
    "19:00", "20:00", "21:00"
]

# 마지막 실행 시간을 저장하여 중복 실행 방지
last_run_date = None



# Upbit 객체 생성
access = os.getenv("UPBIT_ACCESS_KEY")
secret = os.getenv("UPBIT_SECRET_KEY")
if not access or not secret:
    logger.error("API keys not found. Please check your .env file.")
    raise ValueError("Missing API keys. Please check your .env file.")
upbit = pyupbit.Upbit(access, secret)

# OpenAI 구조화된 출력 체크용 클래스
class TradingDecision(BaseModel):
    decision: str
    percentage: int
    reason: str
    reflection: str

# SQLite 데이터베이스 초기화 함수 - 거래 내역을 저장할 테이블을 생성
def init_db():
    db_file = "XRP_trades.db"
    if not os.path.exists(db_file):
        print(f"\n{db_file} 파일이 없습니다. 새로 생성합니다.")
    conn = sqlite3.connect('XRP_trades.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS trades
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  decision TEXT,
                  percentage INTEGER,
                  reason TEXT,
                  XRP_balance REAL,
                  krw_balance REAL,
                  XRP_avg_buy_price REAL,
                  XRP_krw_price REAL,
                  reflection TEXT)''')
    conn.commit()
    return conn

# 거래 기록을 DB에 저장하는 함수
def log_trade(conn, decision, percentage, reason, XRP_balance, krw_balance, XRP_avg_buy_price, XRP_krw_price, reflection):
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute("""INSERT INTO trades 
                 (timestamp, decision, percentage, reason, XRP_balance, krw_balance, XRP_avg_buy_price, XRP_krw_price, reflection) 
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
              (timestamp, decision, percentage, reason, XRP_balance, krw_balance, XRP_avg_buy_price, XRP_krw_price, reflection))
    conn.commit()


# 최근 투자 기록 조회
def get_recent_trades(conn, days=7):
    c = conn.cursor()
    seven_days_ago = (datetime.now() - timedelta(days=days)).isoformat()
    c.execute("SELECT * FROM trades WHERE timestamp > ? ORDER BY timestamp DESC", (seven_days_ago,))
    columns = [column[0] for column in c.description]
    return pd.DataFrame.from_records(data=c.fetchall(), columns=columns)

# 최근 투자 기록을 기반으로 퍼포먼스 계산 (초기 잔고 대비 최종 잔고)
def calculate_performance(trades_df):
    if trades_df.empty:
        return 0 # 기록이 없을 경우 0%로 설정
    # 초기 잔고 계산 (KRW + XRP * 현재 가격)
    initial_balance = trades_df.iloc[-1]['krw_balance'] + trades_df.iloc[-1]['XRP_balance'] * trades_df.iloc[-1]['XRP_krw_price']
    # 최종 잔고 계산
    final_balance = trades_df.iloc[0]['krw_balance'] + trades_df.iloc[0]['XRP_balance'] * trades_df.iloc[0]['XRP_krw_price']
    return (final_balance - initial_balance) / initial_balance * 100

# AI 모델을 사용하여 최근 투자 기록과 시장 데이터를 기반으로 분석 및 반성을 생성하는 함수
def generate_reflection(trades_df, current_market_data):
    performance = calculate_performance(trades_df) # 투자 퍼포먼스 계산
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not client.api_key:
        logger.error("OpenAI API key is missing or invalid.")
        return None
    
    # OpenAI API 호출로 AI의 반성 일기 및 개선 사항 생성 요청
    time.sleep(5)
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": "You are an AI trading assistant tasked with analyzing recent trading performance and current market conditions to generate insights and improvements for future trading decisions."
            },
            {
                "role": "user",
                "content": f"""
                Recent trading data:
                {trades_df.to_json(orient='records')}
                
                Current market data:
                {current_market_data}
                
                Overall performance in the last 7 days: {performance:.2f}%
                
                Please analyze this data and provide:
                1. A brief reflection on the recent trading decisions
                2. Insights on what worked well and what didn't
                3. Suggestions for improvement in future trading decisions
                4. Any patterns or trends you notice in the market data
                
                Limit your response to 100 words or less.
                """
            }
        ]
    )

    try:
        response_content = response.choices[0].message.content
        return response_content
    except (IndexError, AttributeError) as e:
        logger.error(f"Error extracting response content: {e}")
        return None

# 데이터프레임에 보조 지표를 추가하는 함수
def add_indicators(df):
    # 볼린저 밴드 추가
    indicator_bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_bbm'] = indicator_bb.bollinger_mavg()
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()
    
    # RSI (Relative Strength Index) 추가
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    
    # MACD (Moving Average Convergence Divergence) 추가
    macd = ta.trend.MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # 이동평균선 (단기, 장기)
    df['sma_20'] = ta.trend.SMAIndicator(close=df['close'], window=20).sma_indicator()
    df['ema_12'] = ta.trend.EMAIndicator(close=df['close'], window=12).ema_indicator()
    
    return df

# 공포 탐욕 지수 조회
def get_fear_and_greed_index():
    url = "https://api.alternative.me/fng/"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data['data'][0]
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Fear and Greed Index: {e}")
        return None

#뉴스 데이터 가져오기
def get_XRP_news():
    serpapi_key = os.getenv("SERPAPI_API_KEY")
    if not serpapi_key:
        logger.error("SERPAPI API key is missing.")
        return None  # 또는 함수 종료
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_news",
        "q": "XRP",
        "api_key": serpapi_key
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        news_results = data.get("news_results", [])
        headlines = []
        for item in news_results:
            headlines.append({
                "title": item.get("title", ""),
                "date": item.get("date", "")
            })
        
        return headlines[:5]
    except requests.RequestException as e:
        logger.error(f"Error fetching news: {e}")
        return []
        
# 유튜브 자막 데이터 가져오기
def get_combined_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko'])
        combined_text = ' '.join(entry['text'] for entry in transcript)
        return combined_text
    except Exception as e:
        logger.error(f"Error fetching YouTube transcript: {e}")
        return ""

#### Selenium 관련 함수
def create_driver():
    
    env = os.getenv("ENVIRONMENT")
    #print("ChromeDriver 설정 중...\n")
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
        #  WebGL 오류 해결을 위한 추가 옵션
    chrome_options.add_argument("--use-gl=swiftshader")  # 소프트웨어 렌더링을 강제 활성화
    chrome_options.add_argument("--enable-unsafe-webgl")  # WebGL을 강제 활성화
    chrome_options.add_argument("--enable-unsafe-swiftshader")  # SwiftShader 활성화
    
    
    try:
        if env == "local":
            chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
            from webdriver_manager.chrome import ChromeDriverManager
            service = Service(ChromeDriverManager().install())
            
        elif env == "ec2":
            service = Service('/usr/bin/chromedriver')
        else:
            raise ValueError(f"Unsupported environment. Only local or ec2: {env}")
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver
    
    except Exception as e:
        logger.error(f"ChromeDriver 생성 중 오류 발생: {e}")
        raise

# XPath로 Element 찾기
def click_element_by_xpath(driver, xpath, element_name, wait_time=10):
    try:
        element = WebDriverWait(driver, wait_time).until(
            EC.presence_of_element_located((By.XPATH, xpath))
        )
        # 요소가 뷰포트에 보일 때까지 스크롤
        driver.execute_script("arguments[0].scrollIntoView(true);", element)
        # 요소가 클릭 가능할 때까지 대기
        element = WebDriverWait(driver, wait_time).until(
            EC.element_to_be_clickable((By.XPATH, xpath))
        )
        element.click()
        #print(f"{element_name} 클릭 완료")
        time.sleep(2)  # 클릭 후 잠시 대기
    except TimeoutException:
        logger.error(f"{element_name} 요소를 찾는 데 시간이 초과되었습니다.")
    except ElementClickInterceptedException:
        logger.error(f"{element_name} 요소를 클릭할 수 없습니다. 다른 요소에 가려져 있을 수 있습니다.")
    except NoSuchElementException:
        logger.error(f"{element_name} 요소를 찾을 수 없습니다.")
    except Exception as e:
        logger.error(f"{element_name} 클릭 중 오류 발생: {e}")
# 차트 클릭하기
def perform_chart_actions(driver):
    # 시간 메뉴 클릭
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[1]",
        "시간 메뉴"
    )
    # 1시간 옵션 선택
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[1]/cq-menu-dropdown/cq-item[8]",
        "1시간 옵션"
    )
    # 지표 메뉴 클릭
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]",
        "지표 메뉴"
    )
    # 볼린저 밴드 옵션 선택
    click_element_by_xpath(
        driver,
        "/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]/cq-menu-dropdown/cq-scroll/cq-studies/cq-studies-content/cq-item[15]",
        "볼린저 밴드 옵션"
    )
# 스크린샷 캡쳐 및 base64 이미지 인코딩
def capture_and_encode_screenshot(driver):
    try:
        # 스크린샷 캡처
        png = driver.get_screenshot_as_png()
        # PIL Image로 변환
        img = Image.open(io.BytesIO(png))
        # 이미지가 클 경우 리사이즈 (OpenAI API 제한에 맞춤)
        img.thumbnail((2000, 2000))
        # 이미지를 바이트로 변환
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        # base64로 인코딩
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return base64_image
    except Exception as e:
        logger.error(f"스크린샷 캡처 및 인코딩 중 오류 발생: {e}")
        return None

############################################## 메인 AI 트레이딩 로직#################################
def ai_trading():
    print("\n######################### AI Trading #########################\n")
    # tqdm 진행률 바 설정 (경과 시간 표시)
    with tqdm(total=100, desc="####### AI Decision Processing", bar_format="{l_bar}{bar} {n_fmt}/{total_fmt} #######", leave=False) as pbar:
        for i in range(1):  # 10번 진행
            
                for _ in range(1):  # 1초 동안 1초씩 진행
                    time.sleep(1)  # 1초 대기
                    pbar.update(5)  # 1% 증가 (매초 업데이트)                                     #5%
                
                ### 데이터 가져오기
                # 1. 현재 투자 상태 조회
                all_balances = upbit.get_balances()
                filtered_balances = [balance for balance in all_balances if balance['currency'] in ['XRP', 'KRW']]
                
                
                
                
                
                
                # 2. 오더북(호가 데이터) 조회
                orderbook = pyupbit.get_orderbook("KRW-XRP")
                pbar.update(5)  # 2%씩 증가                                     #10%
                
                
                
                # 3. 차트 데이터 조회 및 보조지표 추가
                df_daily = pyupbit.get_ohlcv("KRW-XRP", interval="day", count=30)
                df_daily = dropna(df_daily)
                df_daily = add_indicators(df_daily)
                
                pbar.update(5)  # 5%씩 증가                                     #15%
                
                
                
                df_hourly = pyupbit.get_ohlcv("KRW-XRP", interval="minute60", count=24)
                df_hourly = dropna(df_hourly)
                df_hourly = add_indicators(df_hourly)
                
                
                
                
                
    
                # 4. 공포 탐욕 지수 가져오기
                fear_greed_index = get_fear_and_greed_index()
                
                
                
    
                # 5. 뉴스 헤드라인 가져오기
                news_headlines = get_XRP_news()
                
                for _ in range(1):  # 1초 동안 1초씩 진행
                    time.sleep(1)  # 1초 대기
                    pbar.update(5)  # 1% 증가 (매초 업데이트)                                     #20%
                
                
    
                # 6. YouTube 자막 데이터 가져오기
                # youtube_transcript = get_combined_transcript("3XbtEX3jUv4")
                f = open("strategy.txt", "r", encoding="utf-8")
                youtube_transcript = f.read()
                f.close()
                
                pbar.update(5)  # 5%씩 증가                                     #25%
                
    
                # 7. Selenium으로 차트 캡처
                driver = None
                try:
                    driver = create_driver()
                    driver.get("https://upbit.com/full_chart?code=CRIX.UPBIT.KRW-XRP")
                    #print("페이지 로드 완료")
                    
                
                    for _ in range(6):  # 5초 동안 1초씩 진행
                        time.sleep(5)  # 1초 대기
                        pbar.update(5)  # 1% 증가 (매초 업데이트)                                     #55%
                    
                    
                    #print("차트 작업 시작")
                    
                    perform_chart_actions(driver)
                    #print("차트 작업 완료")
                    
                    chart_image = capture_and_encode_screenshot(driver)
                    #print("스크린샷 캡처 완료.\n")
                    
                except WebDriverException as e:
                    logger.error(f"캡쳐시 WebDriver 오류 발생: {e}")
                    chart_image = None
                except Exception as e:
                    logger.error(f"차트 캡처 중 오류 발생: {e}")
                    chart_image = None
                finally:
                    if driver:
                        driver.quit()
                pbar.update(5)  # 2%씩 증가                                     #60%
                
                        
                
    
                # 8. ### AI에게 데이터 제공하고 판단 받기
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                
                if not client.api_key:
                    logger.error("OpenAI API key is missing or invalid.")
                    return None
                
                try:
                    # 데이터베이스 연결
                    with sqlite3.connect('XRP_trades.db') as conn:
                        #최근 거래 내역 가져오기
                        recent_trades = get_recent_trades(conn)
                        pbar.update(5)  # 2%씩 증가                                     #65%
                        
                        
                        
                        
                        #현재 시장 데이터 수집 (기존 코드에서 가져온 데이터 사용)
                        current_market_data = {
                            "fear_greed_index": fear_greed_index,
                            "news_headlines": news_headlines,
                            "orderbook": orderbook,
                            "daily_ohlcv": df_daily.to_dict(),
                            "hourly_ohlcv": df_hourly.to_dict()
                        }
                        
                        
                        
                        
                        # 반성 및 개선 내용 생성
                        reflection = generate_reflection(recent_trades, current_market_data)
                        for _ in range(1):  # 30초 동안 1초씩 진행
                            time.sleep(5)  # 1초 대기
                            pbar.update(5)  # 1% 증가 (매초 업데이트)                                     #70%
                        
                        
                        
                        
                        # AI 모델에 반성 내용 제공
                        for _ in range(6):  # 30초 동안 1초씩 진행
                            time.sleep(5)  # 1초 대기
                            pbar.update(5)  # 1% 증가 (매초 업데이트)                                     #100%
                        
                            
                        response = client.chat.completions.create(
                            model="gpt-4o-2024-08-06",
                            messages=[
                                {
                                    "role": "system",
                                    "content": """You are an expert in XRP investing. Analyze the provided data...
                                    
                                    - Recent news headlines and their potential impact on XRP price
                                    ### Response Format (MUST follow this format strictly)
                                    - Decision: (buy, sell, or hold)
                                    - Percentage: (1-100 for buy/sell, 0 for hold)
                                    - Reason: (Max 50 characters)
                                    - Reflection: A brief reflection on past trades (Max 100 words)
                                    
                                    """
                                },
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": f"""### Current Investment Status
                            {json.dumps(filtered_balances)}
                        
                            ### Market Data:
                            - Orderbook: {json.dumps(orderbook)}
                            - Daily OHLCV with indicators (30 days): {df_daily.to_json()}
                            - Hourly OHLCV with indicators (24 hours): {df_hourly.to_json()}
                            - Fear and Greed Index: {json.dumps(fear_greed_index)}
                            - Recent news headlines: {json.dumps(news_headlines)}
                            
                            ### Trading History for Last 7 Days:
                            {recent_trades.to_json(orient='records')}
                        
                            ### Trading Strategy Reference:
                            {youtube_transcript}
                            
                            ### Previous Trading Reflection:
                            {reflection}  # 반성 내용 포함
                        
                            Please analyze the above data and provide:
                            - A brief reflection on past trading decisions
                            - Insights on what worked and what didn’t
                            - Suggestions for improvement
                            - A buy/sell/hold decision with reasoning
                            - Recent news headlines and their potential impact on XRP price
                            """
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/png;base64,{chart_image}"
                                            }
                                        }
                                    ]
                                }
                            ],
                            response_format={
                                "type": "json_schema",
                                "json_schema": {
                                    "name": "trading_decision",
                                    "strict": True,
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "decision": {"type": "string", "enum": ["buy", "sell", "hold"]},
                                            "percentage": {"type": "integer"},
                                            "reason": {"type": "string"},
                                            "reflection": {"type": "string"}
                                        },
                                        "required": ["decision", "percentage", "reason", "reflection"],
                                        "additionalProperties": False
                                    }
                                }
                            },
                            max_tokens=2048
                        )
                            
                        
    
                        # 9. # Pydantic을 사용하여 AI의 트레이딩 결정 구조를 정의
                        try:
                            #result = TradingDecision.parse_raw(response.choices[0].message.content) #visual studio
                            result = TradingDecision.model_validate_json(response.choices[0].message.content) #spyder
                            reflection = result.reflection
                            
                        except Exception as e:
                            logger.error(f"Error parsing AI response: {e}")
                            return
                        
                        
                        
                        pbar.close()
                        
                        print(f"\nAI Decision: {result.decision.upper()}")
                        print(f"Decision Reason: {result.reason}\n")
                        
                        # tqdm 진행률 바 설정 (경과 시간 표시)
                        with tqdm(total=100, desc="AI Trading Processing", bar_format="{l_bar}{bar} {n_fmt}/{total_fmt}", leave=False) as pbar:
                            for i in range(1):  # 10번 진행
    
                                order_executed = False
                                pbar.update(35)  # 5%씩 증가
            
                                if result.decision == "buy":
                                    my_krw = upbit.get_balance("KRW")
                                    if my_krw is None:
                                        logger.error("\n\nFailed to retrieve KRW balance.")
                                        return
                                    buy_amount = my_krw * (result.percentage / 100) * 0.9995  # 수수료 고려
                                    if buy_amount > 5000:
                                        print(f"\nBuy Order Executed: {result.percentage}% of available KRW")
                                        try:
                                            order = upbit.buy_market_order("KRW-XRP", buy_amount)
                                            if order:
                                                print(f"\nBuy order executed successfully: {order}")
                                                order_executed = True
                                            else:
                                                logger.error("\nBuy order failed.")
                                        except Exception as e:
                                            logger.error(f"\nError executing buy order: {e}")
                                    else:
                                        logger.warning("\nBuy Order Failed: Insufficient KRW (less than 5000 KRW)\n")
                                elif result.decision == "sell":
                                    my_XRP = upbit.get_balance("KRW-XRP")
                                    if my_XRP is None:
                                        logger.error("\nFailed to retrieve KRW balance.")
                                        return
                                    sell_amount = my_XRP * (result.percentage / 100)
                                    current_price = pyupbit.get_current_price("KRW-XRP")
                                    if sell_amount * current_price > 5000:
                                        print(f"\nSell Order Executed: {result.percentage}% of held XRP")
                                        try:
                                            order = upbit.sell_market_order("KRW-XRP", sell_amount)
                                            if order:
                                                order_executed = True
                                            else:
                                                logger.error("\nBuy order failed.")
                                        except Exception as e:
                                            logger.error(f"\nError executing sell order: {e}")
                                    else:
                                        logger.warning("\nSell Order Failed: Insufficient XRP (less than 5000 KRW worth)\n")
                                
                                
                                
                                # 10. # 거래 실행 여부와 관계없이 현재 잔고 조회
                                for _ in range(8):  # 5초 동안 1초씩 진행
                                    time.sleep(0.625)  # 1초 대기
                                    pbar.update(5)  # 1% 증가 (매초 업데이트)
                                
                                
                                balances = upbit.get_balances()
                                XRP_balance = next((float(balance['balance']) for balance in balances if balance['currency'] == 'XRP'), 0)
                                krw_balance = next((float(balance['balance']) for balance in balances if balance['currency'] == 'KRW'), 0)
                                XRP_avg_buy_price = next((float(balance['avg_buy_price']) for balance in balances if balance['currency'] == 'XRP'), 0)
                                current_XRP_price = pyupbit.get_current_price("KRW-XRP")
                                pbar.update(15)  # 5%씩 증가
                                
            
                                # 11. # 거래 기록을 DB에 저장하기
                                log_trade(conn, result.decision, result.percentage if order_executed else 0, result.reason, 
                                        XRP_balance, krw_balance, XRP_avg_buy_price, current_XRP_price, reflection)
                                
                                time.sleep(2)  # 1초 대기
                                pbar.update(5)  # 1% 증가 (매초 업데이트)
                                pbar.close()
                                
                                        
                                
                                    
                                
                except sqlite3.Error as e:
                            logger.error(f"Database connection error: {e}")
                            return
                        
                time.sleep(5)  # 5초 대기
                    
                
                
                print("\n##################### Waiting for Trading #####################\n")
                
                         
    



if __name__ == "__main__":
    
    conn = init_db()  # <== 이 부분 추가 (DB 초기화)

    while True:
        # 현재 한국 시간 가져오기
        now = datetime.now(KST)
        current_date = now.strftime("%Y-%m-%d")  # 현재 날짜 (예: "2025-01-30")
        current_time = now.strftime("%H:%M")
        current_day = now.strftime("%A")  # 요일 가져오기 ("Monday", "Tuesday", ...)
        
        balances = upbit.get_balances()
        XRP_balance = next((float(balance['balance']) for balance in balances if balance['currency'] == 'XRP'), 0)
        krw_balance = next((float(balance['balance']) for balance in balances if balance['currency'] == 'KRW'), 0)
        XRP_avg_buy_price = next((float(balance['avg_buy_price']) for balance in balances if balance['currency'] == 'XRP'), 0)
        current_XRP_price = pyupbit.get_current_price("KRW-XRP")
        
        total_value = krw_balance+XRP_balance*current_XRP_price
        coin_profit = current_XRP_price/XRP_avg_buy_price*100-100
        coin_value = XRP_balance*current_XRP_price
        #총 자산: {total_value:.2f}원
        print(f"\n\n###################### {current_date} {current_day} {current_time} ######################")
        #print(f"\n### 총 자산: {total_value:,.2f}원, 보유 코인 {XRP_balance:,.5f}개, 코인 가치: {coin_value:,.2f}원, 수익률: {coin_profit:,.2f}% ###\n")
        print(f"\n### 총 자산: {total_value:,.2f}원, 코인 가치: {coin_value:,.2f}원, 수익률: {coin_profit:,.2f}% ###\n")
        print("#####################################################################\n")
        
        time.sleep(30)

        # 날짜가 바뀌었으면 초기화 (새로운 날짜에서 다시 실행할 수 있도록)
        if last_run_date != now.date():
            last_run_date = now.date()
    
        # 실행 시간을 다시 초기화
        EXECUTION_TIMES = [
            "01:00", "05:00", "09:00", "10:00", "11:00", "12:00",
            "13:00", "14:00", "15:00", "16:00", "17:00", "18:00",
            "19:00", "20:00", "21:00"
        ]


        # 실행할 요일 + 실행할 시간 체크
        if current_day in EXECUTION_DAYS and current_time in EXECUTION_TIMES:
            print(f"\n\n################### {current_date} {current_day} {current_time} ###################")
            ai_trading()

            # 실행한 시간을 기록하여 중복 실행 방지
            EXECUTION_TIMES.remove(current_time)  

            # ai_trading 실행 후 날짜 갱신
            last_run_date = now.date() 
        
            # 60초마다 시간 체크 (너무 자주 체크하면 CPU 사용량 증가)
            time.sleep(30)
        
        
        
        