# Purpose: The Sensors
# Key Contents: Custom tools to read your synthetic dataframe and search for news.

from crewai.tools import tool

class SocialMediaTools:
    @tool("DataColumnReader")
    def read_post_data(column_name: str, shared_data: dict) -> str:
        """Reads specific columns from the current social media post record."""
        # This helps the agent pick out just the 'post_text' or 'comments'
        return shared_data.get(column_name, f"Column {column_name} not found.")