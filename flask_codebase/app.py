from flask import Flask, render_template, request, redirect, url_for
import logging
import requests

API_KEY = "da7cc643495745a78c99c491e1d4d0a6"
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)
### Functions here ###

def fetch_historical_data(ticker: str, start_date: str, end_date: str, time_interval: str):
    data_format = "JSON"
    timezone = "utc"
    historical_data_url = f"https://api.twelvedata.com/time_series?apikey={API_KEY}&interval={time_interval}&symbol={ticker}&start_date={start_date} 16:23:00&end_date={end_date} 16:00:00&format={data_format}&timezone={timezone}"
    try:
        response = requests.get(historical_data_url)
        logging.debug(response)
        data = response.json()
        logging.info(data)
        return data
    except Exception as e:
        logging.error(f"Error fetching historical data: {e}")
        return None  # Return None or handle errors as needed
    
### Flask Routing here ###
@app.route('/')
def home():
    return render_template('index.html', fig=None)  # This will render your HTML page

# this is on submit on the search for historical prices
@app.route('/submit-search-historical-prices', methods=['POST'])
def submit_search_historical_prices():
    # Get values from the form in the html page
    symbol = request.form['symbol']
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    time_interval = request.form['time_interval']
    
    # Perform your logic here with the received values
    # For example, you can print them or process them further
    logging.info(f'CCY Pair Symbol: {symbol}, Start Date: {start_date}, End Date: {end_date}, Time Interval: {time_interval}')
    fetch_historical_data(symbol, start_date, end_date, time_interval)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)