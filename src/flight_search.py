"""
Flight search functionality for the Travel Assistant.
Handles filtering and searching through flight data.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from src.utils import load_json_file, parse_date, extract_month_year
from src.config import FLIGHTS_FILE


class FlightSearcher:
    """Handles flight search operations."""
    
    def __init__(self):
        """Initialize the flight searcher with flight data."""
        self.flights = load_json_file(FLIGHTS_FILE)
    
    def search(
        self,
        origin: Optional[str] = None,
        destination: Optional[str] = None,
        departure_month: Optional[int] = None,
        departure_year: Optional[int] = None,
        alliance: Optional[str] = None,
        airline: Optional[str] = None,
        max_price: Optional[float] = None,
        refundable_only: bool = False,
        avoid_overnight_layover: bool = False,
        max_layovers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for flights based on criteria.
        
        Args:
            origin: Departure city
            destination: Arrival city
            departure_month: Month of departure (1-12)
            departure_year: Year of departure
            alliance: Airline alliance (e.g., "Star Alliance")
            airline: Specific airline name
            max_price: Maximum price in USD
            refundable_only: Only show refundable flights
            avoid_overnight_layover: Exclude flights with overnight layovers
            max_layovers: Maximum number of layovers
            
        Returns:
            List of matching flights
        """
        results = self.flights.copy()
        
        # Filter by origin
        if origin:
            origin_lower = origin.lower()
            results = [
                f for f in results 
                if origin_lower in f.get('from', '').lower()
            ]
        
        # Filter by destination
        if destination:
            dest_lower = destination.lower()
            results = [
                f for f in results 
                if dest_lower in f.get('to', '').lower()
            ]
        
        # Filter by departure date
        if departure_month and departure_year:
            results = [
                f for f in results
                if self._matches_date(f.get('departure_date'), departure_month, departure_year)
            ]
        
        # Filter by alliance
        if alliance:
            alliance_lower = alliance.lower()
            results = [
                f for f in results
                if alliance_lower in f.get('alliance', '').lower()
            ]
        
        # Filter by airline
        if airline:
            airline_lower = airline.lower()
            results = [
                f for f in results
                if airline_lower in f.get('airline', '').lower()
            ]
        
        # Filter by price
        if max_price is not None:
            results = [
                f for f in results
                if f.get('price_usd', float('inf')) <= max_price
            ]
        
        # Filter by refundable
        if refundable_only:
            results = [
                f for f in results
                if f.get('refundable', False)
            ]
        
        # Filter by overnight layover
        if avoid_overnight_layover:
            results = [
                f for f in results
                if not f.get('overnight_layover', False)
            ]
        
        # Filter by number of layovers
        if max_layovers is not None:
            results = [
                f for f in results
                if len(f.get('layovers', [])) <= max_layovers
            ]
        
        # Sort by price (ascending)
        results.sort(key=lambda x: x.get('price_usd', float('inf')))
        
        return results
    
    def _matches_date(self, date_str: str, month: int, year: int) -> bool:
        """
        Check if a date string matches the given month and year.
        
        Args:
            date_str: Date string from flight data
            month: Target month (1-12)
            year: Target year
            
        Returns:
            True if date matches, False otherwise
        """
        date_obj = parse_date(date_str)
        if not date_obj:
            return False
        return date_obj.month == month and date_obj.year == year


def create_flight_search_tool():
    """
    Create a flight search tool for LangChain agent.
    
    Returns:
        Tuple of (function, description)
    """
    searcher = FlightSearcher()
    
    def search_flights(query: str) -> str:
        """
        Search for flights based on natural language query.
        
        The query should contain search criteria like:
        - Origin and destination cities
        - Travel dates or months
        - Airline or alliance preferences
        - Price constraints
        - Layover preferences
        
        Args:
            query: Natural language search query
            
        Returns:
            Formatted string with flight results
        """
        # Parse query to extract criteria
        query_lower = query.lower()
        
        # Extract destination and origin
        origin = None
        destination = None
        
        # Common origin cities
        if 'from dubai' in query_lower or 'dubai to' in query_lower:
            origin = 'Dubai'
        
        # Extract destination
        destinations = ['tokyo', 'paris', 'london', 'new york', 'bangkok', 'hong kong', 
                       'seoul', 'taipei', 'toronto', 'helsinki', 'zurich', 'kuala lumpur']
        for dest in destinations:
            if dest in query_lower:
                destination = dest.title()
                break
        
        # Extract month/year
        month, year = extract_month_year(query)
        
        # Extract alliance
        alliance = None
        if 'star alliance' in query_lower:
            alliance = 'Star Alliance'
        elif 'oneworld' in query_lower:
            alliance = 'Oneworld'
        elif 'skyteam' in query_lower:
            alliance = 'SkyTeam'
        
        # Extract price constraint
        max_price = None
        if 'under' in query_lower or 'less than' in query_lower or 'below' in query_lower:
            # Try to extract price
            import re
            price_match = re.search(r'\$?(\d{3,4})', query)
            if price_match:
                max_price = float(price_match.group(1))
        
        # Check for refundable requirement
        refundable_only = 'refundable' in query_lower
        
        # Check for layover preferences
        avoid_overnight = 'avoid overnight' in query_lower or 'no overnight' in query_lower
        
        max_layovers = None
        if 'direct' in query_lower or 'non-stop' in query_lower or 'nonstop' in query_lower:
            max_layovers = 0
        elif '1 layover' in query_lower or 'one layover' in query_lower:
            max_layovers = 1
        
        # Perform search
        results = searcher.search(
            origin=origin,
            destination=destination,
            departure_month=month,
            departure_year=year,
            alliance=alliance,
            max_price=max_price,
            refundable_only=refundable_only,
            avoid_overnight_layover=avoid_overnight,
            max_layovers=max_layovers
        )
        
        # Format results
        from src.utils import format_flights_list
        return format_flights_list(results, max_results=5)
    
    return search_flights