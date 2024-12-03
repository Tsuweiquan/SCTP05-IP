window.addEventListener("DOMContentLoaded", async function () {
    let BASE_API_URL = 'https://api.twelvedata.com';
    let API_KEY = 'da7cc643495745a78c99c491e1d4d0a6';
    let FORMAT = 'JSON';

    let broker_API_KEY = '5323dbb8a95989f8308e2367421ce6c9-459c4f4988410858172c9ccfa1258fa7';
    let BROKER_API_URL = '';

    let time_API_KEY = '38ea3a8b5fffc9152b3e0d9b517bd8eb';
    let time_API_URL = 'https://api.api-ninjas.com/v1/worldtime';

    document.querySelector("#search-button").addEventListener("click", async function () {
        let INTERVAL = document.querySelector("#timeinterval").value;
        let SYMBOL = document.querySelector("#search-pair").value;
        let START_DATE = document.querySelector("#startDate").value;
        let END_DATE = document.querySelector("#endDate").value;

        // Convert the date format to yyyy/mm/dd
        let formattedStartDate = START_DATE.replace(/-/g, '/'); // Replace '-' with '/' to get yyyy/mm/dd
        let formattedEndDate = END_DATE.replace(/-/g, '/'); // Replace '-' with '/' to get yyyy/mm/dd

        // Make the API request using Axios
        let response = await axios.get(`${BASE_API_URL}/time_series?apikey=${API_KEY}&interval=${INTERVAL}&start_date=${formattedStartDate}&end_date=${formattedEndDate}&format=${FORMAT}&symbol=${SYMBOL}`)
        let allCurrencyUsd = response.data.values;


        // Function to render the data in the table
        function renderTasks() {
            let tbody = document.querySelector("#priceTable tbody");
            tbody.innerHTML = ""; // Clear any previous rows

            for (let i = 0; i < allCurrencyUsd.length; i++) {
                let { datetime, open, high, low, close } = allCurrencyUsd[i];

                // Create a new row
                let row = `<tr>
                            <td>${datetime}</td>
                            <td>${open}</td>
                            <td>${high}</td>
                            <td>${low}</td>
                            <td>${close}</td>
                           </tr>`;

                // Append the row to the table body
                tbody.innerHTML += row;
            }
        }
        // Call the function to render the data
        renderTasks();
    });



    async function weatherTask() {
        let weather_Url = "http://api.weatherapi.com/v1/current.json?key=d0145b08e27343808f8151015242710&q=Singapore&aqi=no";
        let response = await axios.get(weather_Url);
        console.log(response.data)

        let city = response.data.location.country;
        console.log(city);

        document.querySelector('.city').innerHTML = response.data.location.country;
        document.querySelector('.Temp').innerHTML = response.data.current.temp_c + "°C";
        document.querySelector('.weather img').src = "https:" + response.data.current.condition.icon;
    }

 
    weatherTask();

});

