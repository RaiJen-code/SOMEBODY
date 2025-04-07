"""
Internet search utilities for SOMEBODY
"""

import logging
import os
import json
import requests
import wikipedia
from typing import Dict, List, Optional, Any
from googleapiclient.discovery import build

logger = logging.getLogger(__name__)

class InternetSearch:
    """Internet search capabilities using various APIs"""
    
    def __init__(self, google_api_key: Optional[str] = None, 
                 google_cse_id: Optional[str] = None):
        """Initialize internet search
        
        Args:
            google_api_key: Google API key for search
            google_cse_id: Google Custom Search Engine ID
        """
        self.google_api_key = google_api_key or os.environ.get("GOOGLE_API_KEY")
        self.google_cse_id = google_cse_id or os.environ.get("GOOGLE_CSE_ID")
        self.google_service = None
        
        if self.google_api_key and self.google_cse_id:
            try:
                self.google_service = build(
                    "customsearch", "v1", 
                    developerKey=self.google_api_key
                )
                logger.info("Google Custom Search initialized")
            except Exception as e:
                logger.error(f"Error initializing Google Custom Search: {str(e)}")
        else:
            logger.warning("Google API key or CSE ID not provided, Google search will be disabled")
            
    def google_search(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """Perform Google search
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of search results (title, link, snippet)
        """
        if not self.google_service:
            logger.warning("Google search attempted but Google API not initialized")
            return []
            
        try:
            logger.info(f"Performing Google search for: {query}")
            
            # Execute search
            res = self.google_service.cse().list(
                q=query,
                cx=self.google_cse_id,
                num=num_results
            ).execute()
            
            # Parse results
            search_results = []
            if "items" in res:
                for item in res["items"]:
                    search_results.append({
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", "")
                    })
                    
            logger.info(f"Google search returned {len(search_results)} results")
            return search_results
        except Exception as e:
            logger.error(f"Error in Google search: {str(e)}")
            return []
            
    def wikipedia_search(self, query: str, sentences: int = 3) -> Dict[str, Any]:
        """Search Wikipedia for information
        
        Args:
            query: Search query
            sentences: Number of sentences to return in summary
            
        Returns:
            Dictionary with search results
        """
        try:
            logger.info(f"Searching Wikipedia for: {query}")
            
            # Set language
            wikipedia.set_lang("id")  # Indonesian, can be changed as needed
            
            # Search for pages
            search_results = wikipedia.search(query, results=5)
            
            if not search_results:
                logger.info(f"No Wikipedia results found for: {query}")
                return {"success": False, "message": "No results found"}
                
            # Try to get a page summary
            try:
                # Get the first search result's page
                page = wikipedia.page(search_results[0])
                
                # Get a summary
                summary = wikipedia.summary(search_results[0], sentences=sentences)
                
                return {
                    "success": True,
                    "title": page.title,
                    "summary": summary,
                    "url": page.url,
                    "search_results": search_results
                }
            except wikipedia.DisambiguationError as e:
                # Handle disambiguation pages
                logger.info(f"Wikipedia disambiguation for: {query}")
                return {
                    "success": True,
                    "title": f"Multiple results for {query}",
                    "summary": f"Your query has multiple possibilities: {', '.join(e.options[:5])}",
                    "disambiguation": e.options[:5],
                    "search_results": search_results
                }
            except Exception as e:
                logger.error(f"Error getting Wikipedia page: {str(e)}")
                return {
                    "success": False,
                    "message": str(e),
                    "search_results": search_results
                }
                
        except Exception as e:
            logger.error(f"Error in Wikipedia search: {str(e)}")
            return {"success": False, "message": str(e)}
            
    def search(self, query: str) -> Dict[str, Any]:
        """Perform comprehensive internet search
        
        Args:
            query: Search query
            
        Returns:
            Dictionary with combined search results
        """
        results = {
            "query": query,
            "wikipedia": None,
            "google": None
        }
        
        # Try Wikipedia first
        wiki_results = self.wikipedia_search(query)
        results["wikipedia"] = wiki_results
        
        # Then try Google if available
        if self.google_service:
            google_results = self.google_search(query)
            results["google"] = google_results
        
        return results
            
    def get_current_weather(self, location: str) -> Dict[str, Any]:
        """Get current weather for a location
        
        Args:
            location: Location name
            
        Returns:
            Weather information
        """
        try:
            # Using OpenWeatherMap API as an example
            # You would need to sign up for an API key at openweathermap.org
            api_key = os.environ.get("OPENWEATHER_API_KEY")
            if not api_key:
                return {"success": False, "message": "Weather API key not configured"}
                
            # Make API request
            url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                
                weather_info = {
                    "success": True,
                    "location": data["name"],
                    "country": data["sys"]["country"],
                    "weather": data["weather"][0]["main"],
                    "description": data["weather"][0]["description"],
                    "temperature": data["main"]["temp"],
                    "feels_like": data["main"]["feels_like"],
                    "humidity": data["main"]["humidity"],
                    "wind_speed": data["wind"]["speed"]
                }
                
                return weather_info
            else:
                return {"success": False, "message": f"Error: {response.status_code} - {response.text}"}
                
        except Exception as e:
            logger.error(f"Error getting weather: {str(e)}")
            return {"success": False, "message": str(e)}