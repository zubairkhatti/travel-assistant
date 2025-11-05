"""
Utility functions for the Travel Assistant.
"""

import json
from datetime import datetime
from typing import List, Dict, Any
from dateutil import parser


def load_json_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Load and parse a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        List of dictionaries containing the data
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {filepath}: {str(e)}")


def load_text_file(filepath: str) -> str:
    """
    Load a text file.
    
    Args:
        filepath: Path to the text file
        
    Returns:
        String containing file contents
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")


def parse_date(date_str: str) -> datetime:
    """
    Parse a date string flexibly.
    
    Args:
        date_str: Date string in various formats
        
    Returns:
        datetime object
    """
    try:
        return parser.parse(date_str)
    except (ValueError, TypeError):
        return None


def format_flight(flight: Dict[str, Any]) -> str:
    """
    Format a flight dictionary into a readable string.
    
    Args:
        flight: Flight dictionary
        
    Returns:
        Formatted string representation
    """
    layover_str = ""
    if flight.get("layovers"):
        layovers = ", ".join(flight["layovers"])
        overnight = " (overnight)" if flight.get("overnight_layover") else ""
        layover_str = f"\n  Layovers: {layovers}{overnight}"
    
    refundable = "Yes" if flight.get("refundable") else "No"
    
    return f"""
Flight {flight.get('flight_id', 'N/A')}:
  Airline: {flight.get('airline', 'N/A')} ({flight.get('alliance', 'N/A')})
  Route: {flight.get('from', 'N/A')} → {flight.get('to', 'N/A')}{layover_str}
  Dates: {flight.get('departure_date', 'N/A')} to {flight.get('return_date', 'N/A')}
  Price: ${flight.get('price_usd', 'N/A')} USD
  Refundable: {refundable}
"""


def format_flights_list(flights: List[Dict[str, Any]], max_results: int = 5) -> str:
    """
    Format a list of flights into a readable string.
    
    Args:
        flights: List of flight dictionaries
        max_results: Maximum number of results to display
        
    Returns:
        Formatted string with all flights
    """
    if not flights:
        return "No flights found matching your criteria."
    
    result = f"Found {len(flights)} flight(s):\n"
    result += "=" * 50 + "\n"
    
    for flight in flights[:max_results]:
        result += format_flight(flight)
        result += "-" * 50 + "\n"
    
    if len(flights) > max_results:
        result += f"\n(Showing top {max_results} of {len(flights)} results)"
    
    return result


def extract_month_year(text: str) -> tuple:
    """
    Extract month and year from natural language text.
    
    Args:
        text: Text containing date reference
        
    Returns:
        Tuple of (month, year) or (None, None)
    """
    text_lower = text.lower()
    months = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    
    current_year = datetime.now().year
    
    for month_name, month_num in months.items():
        if month_name in text_lower:
            return month_num, current_year
    
    return None, None