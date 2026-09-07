import requests
import json
from rich.panel import Panel
from ..proxies import proxies

GET_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather and next 12 hours forecast for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["city"]
        }
    }
}

def get_weather(city: str,
                cancellation_token=None,) -> str:
    #console.rule("Step 2 - Weather Tool")

    if cancellation_token:
        cancellation_token.raise_if_cancelled()

    geo_response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "en", "format": "json"},
        proxies=proxies,
        verify=False, # for enabling proxyman requests
        timeout=30,
    )

    if cancellation_token:
        cancellation_token.raise_if_cancelled()

    geo_response.raise_for_status()
    geo_data = geo_response.json()

    if not geo_data.get("results"):
        result = f"No location found for city: {city}"
        #console.print(Panel(result, title="Weather Result"))
        return result

    location = geo_data["results"][0]
    latitude = location["latitude"]
    longitude = location["longitude"]

    if cancellation_token:
        cancellation_token.raise_if_cancelled()

    weather_response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,precipitation,rain,weather_code,wind_speed_10m",
            "hourly": "temperature_2m,precipitation_probability,rain",
            "forecast_days": 2,
            "timezone": "auto",
        },
        proxies=proxies,
        verify=False, # for enabling proxyman requests
        timeout=30,
    )

    if cancellation_token:
        cancellation_token.raise_if_cancelled()

    weather_response.raise_for_status()
    weather_data = weather_response.json()

    hourly = weather_data.get("hourly", {})
    next_12_hours = []

    times = hourly.get("time", [])[:12]
    temps = hourly.get("temperature_2m", [])[:12]
    rain_probs = hourly.get("precipitation_probability", [])[:12]
    rains = hourly.get("rain", [])[:12]

    for i, time_value in enumerate(times):
        next_12_hours.append(
            {
                "time": time_value,
                "temperature_c": temps[i] if i < len(temps) else None,
                "rain_probability_percent": rain_probs[i] if i < len(rain_probs) else None,
                "rain_mm": rains[i] if i < len(rains) else None,
            }
        )

    result = {
        "location": {
            "name": location.get("name", city),
            "region": location.get("admin1", ""),
            "country": location.get("country", ""),
            "latitude": latitude,
            "longitude": longitude,
        },
        "current": weather_data.get("current", {}),
        "next_12_hours": next_12_hours,
    }

    result_text = json.dumps(result, indent=2, ensure_ascii=False)
    #console.print(Panel(result_text, title="Weather API Result"))

    return result_text
