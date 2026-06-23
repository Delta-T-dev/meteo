import anthropic
import requests
import json
import math
import os

client = anthropic.Anthropic()

tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city using Open-Meteo API",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
        }
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression to evaluate"}
            },
            "required": ["expression"]
        }
    },
    {
        "name": "save_to_file",
        "description": "Save text content to a file",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "File name"},
                "content": {"type": "string", "description": "Content to save"}
            },
            "required": ["filename", "content"]
        }
    }
]


MOCK_WEATHER = {
    "paris": {"city": "Paris", "country": "France", "temperature_c": 18.4, "humidity_pct": 72, "wind_speed_kmh": 14.2, "condition": "Partly cloudy"},
    "tokyo": {"city": "Tokyo", "country": "Japan", "temperature_c": 29.1, "humidity_pct": 85, "wind_speed_kmh": 8.7, "condition": "Humid and sunny"},
    "berlin": {"city": "Berlin", "country": "Germany", "temperature_c": 21.3, "humidity_pct": 60, "wind_speed_kmh": 18.5, "condition": "Mostly sunny"},
    "madrid": {"city": "Madrid", "country": "Spain", "temperature_c": 34.7, "humidity_pct": 28, "wind_speed_kmh": 11.0, "condition": "Hot and dry"},
    "london": {"city": "London", "country": "United Kingdom", "temperature_c": 16.2, "humidity_pct": 78, "wind_speed_kmh": 22.0, "condition": "Overcast"},
    "new york": {"city": "New York", "country": "United States", "temperature_c": 26.5, "humidity_pct": 65, "wind_speed_kmh": 16.3, "condition": "Partly cloudy"},
    "sydney": {"city": "Sydney", "country": "Australia", "temperature_c": 14.8, "humidity_pct": 68, "wind_speed_kmh": 20.1, "condition": "Cloudy"},
    "dubai": {"city": "Dubai", "country": "UAE", "temperature_c": 41.2, "humidity_pct": 45, "wind_speed_kmh": 13.5, "condition": "Sunny and very hot"},
}

def get_weather(city: str) -> dict:
    key = city.lower().strip()
    if key in MOCK_WEATHER:
        return MOCK_WEATHER[key]
    # Try live Open-Meteo (may be blocked in restricted environments)
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1}, timeout=5
        ).json()
        if not geo.get("results"):
            return {"error": f"City '{city}' not found"}
        loc = geo["results"][0]
        weather = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                "timezone": "auto"
            }, timeout=5
        ).json()
        current = weather["current"]
        return {
            "city": loc["name"],
            "country": loc.get("country", ""),
            "temperature_c": current["temperature_2m"],
            "humidity_pct": current["relative_humidity_2m"],
            "wind_speed_kmh": current["wind_speed_10m"],
        }
    except Exception as e:
        return {"error": f"Weather API unavailable: {e}. Note: using mock data only for Paris and Tokyo."}


def calculate(expression: str) -> dict:
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return {"result": result, "expression": expression}
    except Exception as e:
        return {"error": str(e)}


def save_to_file(filename: str, content: str) -> dict:
    path = os.path.join("/home/user/meteo", filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return {"saved": path, "bytes": len(content.encode())}


def run_tool(name: str, inputs: dict) -> str:
    if name == "get_weather":
        result = get_weather(**inputs)
    elif name == "calculate":
        result = calculate(**inputs)
    elif name == "save_to_file":
        result = save_to_file(**inputs)
    else:
        result = {"error": f"Unknown tool: {name}"}
    return json.dumps(result, ensure_ascii=False)


def run_agent(user_message: str, on_event=None):
    def emit(event: dict):
        if on_event:
            on_event(event)

    print(f"\nUser: {user_message}\n")
    messages = [{"role": "user", "content": user_message}]
    iteration = 0

    while True:
        iteration += 1
        emit({"type": "iteration", "number": iteration})

        response = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=4096,
            tools=tools,
            messages=messages,
            thinking={"type": "adaptive"}
        )

        messages.append({"role": "assistant", "content": response.content})

        for block in response.content:
            if block.type == "thinking" and getattr(block, "thinking", None):
                print(f"[Thinking] {block.thinking[:200]}...")
                emit({"type": "thinking", "content": block.thinking})

        if response.stop_reason != "tool_use":
            final_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_text += block.text
            print(f"Agent: {final_text}")
            emit({"type": "final", "content": final_text})
            break

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"[Tool] {block.name}({json.dumps(block.input, ensure_ascii=False)})")
                emit({"type": "tool_call", "name": block.name, "input": block.input})
                output = run_tool(block.name, block.input)
                print(f"[Result] {output}\n")
                emit({"type": "tool_result", "name": block.name, "result": output})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output
                })

        messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    print("🤖 Agent météo prêt. Tape ta demande (ou 'quitter' pour sortir).\n")
    while True:
        objectif = input("Toi > ")
        if objectif.lower() in ("quitter", "exit", "q"):
            break
        run_agent(objectif)
        print()
