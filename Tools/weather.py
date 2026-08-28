import logging
import requests
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    61: "light rain", 63: "moderate rain", 65: "heavy rain",
    71: "light snow", 73: "moderate snow", 75: "heavy snow",
    80: "light rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}


@tool
def get_weather(location: str) -> str:
    """
    Get the current weather for a given city or location name.
    """
    try:
        geo_resp = requests.get(
            GEOCODE_URL,
            params={"name": location, "count": 1},
            timeout=5,
        )
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()

        results = geo_data.get("results")
        if not results:
            return f"Could not find a location matching '{location}'."

        place = results[0]
        lat, lon = place["latitude"], place["longitude"]
        resolved_name = place.get("name", location)

        weather_resp = requests.get(
            WEATHER_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            },
            timeout=5,
        )
        weather_resp.raise_for_status()
        current = weather_resp.json().get("current", {})

        if not current:
            return f"Weather data is currently unavailable for {resolved_name}."

        temp = current.get("temperature_2m")
        humidity = current.get("relative_humidity_2m")
        wind = current.get("wind_speed_10m")
        code = current.get("weather_code")
        condition = WEATHER_CODES.get(code, "unknown conditions")

        return (
            f"Current weather in {resolved_name}: {condition}, "
            f"{temp}°C, humidity {humidity}%, wind {wind} km/h."
        )

    except Exception as e:
        logger.warning("Weather lookup failed for '%s': %s", location, e)
        return (
            "Weather lookup is temporarily unavailable for this location. "
            "Please try again."
        )


weather_tools = [get_weather]