import cohere
import os

co = cohere.Client(api_key="<YOUR-COHERE-KEY>")
import json

def list_calendar_events(date: str):
  events = '[{"start": "14:00", "end": "15:00"}, {"start": "15:00", "end": "16:00"}, {"start": "17:00", "end": "18:00"}]'
  print(f"Listing events: {events}")
  return events

def create_calendar_event(date: str, time: str, duration: int):
  print(f"Creating a {duration} hour long event at {time} on {date}")
  return True


list_calendar_events_tool = {
  "name": "list_calendar_events",
  "description": "returns a list of calendar events for the specified date, including the start time and end time for each event",
  "parameter_definitions": {
    "date": {
      "description": "the date to list events for, formatted as mm/dd/yy",
      "type": "str",
      "required": True
    }
  }
}

create_calendar_event_tool = {
  "name": "create_calendar_event_tool",
  "description": "creates a calendar event of the specified duration at the specified time and date",
  "parameter_definitions": {
    "date": {
      "description": "the date on which the event starts, formatted as mm/dd/yy",
      "type": "str",
      "required": True
    },
    "time": {
      "description": "the time of the event, formatted using 24h military time formatting",
      "type": "str",
      "required": True
    },
    "duration": {
      "description": "the number of hours the event lasts for",
      "type": "float",
      "required": True
    }
  }
}

def invoke_tool(tool_call: cohere.ToolCall):
  if tool_call.name == list_calendar_events_tool["name"]:
    date = tool_call.parameters["date"]
    return [{
        "events": list_calendar_events(date)
    }]
  elif tool_call.name == create_calendar_event_tool["name"]:
    date = tool_call.parameters["date"]
    time = tool_call.parameters["time"]
    duration = tool_call.parameters["duration"]

    return [{
        "is_success": create_calendar_event(date, time, duration)
    }]
  else:
    raise f"Unknown tool name '{tool_call.name}'"


# Check what tools the model wants to use and how to use them
res = co.chat(
    model="command-r-plus",
    preamble="Today is Thursday, may 23, 2024",
    message="book an hour long appointment for the first available free slot after 3pm",
    force_single_step=False,
    tools=[list_calendar_events_tool, create_calendar_event_tool])


while res.tool_calls:
  print(res.text) # This will be an observation and a plan with next steps

  # invoke the recommended tools
  tool_results = []
  for call in res.tool_calls:
    tool_results.append({"call": call, "outputs": invoke_tool(call)})

  # send back the tool results
  res = co.chat(
    model="command-r-plus",
    chat_history=res.chat_history,
    message="",
    force_single_step=False,
    tools=[list_calendar_events_tool, create_calendar_event_tool],
    tool_results=tool_results,
  )

print(res.text) # print the final answer