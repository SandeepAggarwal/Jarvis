import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, WebDriverException
import time
import os

# Chrome options for headless mode
CHROME_OPTIONS = Options()
CHROME_OPTIONS.add_argument('--headless')
CHROME_OPTIONS.add_argument('--no-sandbox')
CHROME_OPTIONS.add_argument('--disable-dev-shm-usage')

# Chrome driver path (will be auto-detected or use default)
CHROME_DRIVER_PATH = os.environ.get('CHROME_DRIVER_PATH', None)

WEB_AUTOMATION_TOOL = {
    "type": "function",
    "function": {
        "name": "web_automation",
        "description": "Navigate to a webpage and perform actions like clicking buttons. Returns the page content after actions.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to navigate to."
                },
                "action": {
                    "type": "string",
                    "description": "The action to perform (e.g., 'click', 'navigate', 'get_content'). For 'click', provide 'element_selector'."
                },
                "element_selector": {
                    "type": "string",
                    "description": "CSS selector or XPath to locate the element to click. For 'click' action."
                },
                "wait_seconds": {
                    "type": "number",
                    "description": "Seconds to wait before performing action (default: 5)."
                }
            },
            "required": ["url", "action"]
        }
    }
}


def web_automation(url: str, 
                    action: str, 
                    element_selector: str = None, 
                    wait_seconds: float = 5.0,
                    cancellation_token=None,) -> str:
    """
    Navigate to a webpage and perform actions.
    
    Args:
        url: The URL to navigate to
        action: The action to perform ('click', 'navigate', 'get_content')
        element_selector: CSS selector or XPath for the element to click (for 'click' action)
        wait_seconds: Seconds to wait before performing action
    
    Returns:
        Page content after actions
    """
    try:
        # Initialize Chrome driver
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        
        # Try to find ChromeDriver
        chrome_driver_path = None
        for path in [
            '/usr/bin/chromedriver',
            '/usr/local/bin/chromedriver',
            '/opt/chrome/chromedriver',
            os.environ.get('CHROME_DRIVER_PATH'),
        ]:
            if not path:
                continue
            if os.path.exists(path):
                chrome_driver_path = path
                break
        
        if not chrome_driver_path:
            # Try to download or use default
            chrome_driver_path = 'chromedriver'
        
        driver = webdriver.Chrome(options=chrome_options)
        
        # Navigate to URL
        driver.get(url)
        
        # Wait for page to load
        time.sleep(wait_seconds)

        if cancellation_token:
            cancellation_token.raise_if_cancelled()
        
        # Perform action
        if action == 'click':
            if element_selector:
                try:
                    WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, element_selector))
                    ).click()
                    print(f"Clicked element with selector: {element_selector}")
                    if cancellation_token:
                        cancellation_token.raise_if_cancelled()
                except TimeoutException:
                    print(f"Element not found with selector: {element_selector}")
                    return f"Error: Element not found with selector '{element_selector}'"
            else:
                print("No element selector provided for click action")
                return "Error: element_selector required for click action"
        elif action == 'navigate':
            print(f"Navigated to: {url}")
        elif action == 'get_content':
            print(f"Getting content from: {url}")
        
        # Get page content
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        if cancellation_token:
            cancellation_token.raise_if_cancelled()
        
        # Clean up the page
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript']):
            tag.decompose()
        
        text = soup.get_text(separator='\n')
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        
        # Limit output
        result = '\n'.join(lines[:5000])
        
        # Close driver
        driver.quit()
        
        return result
    
    except Exception as e:
        return f"Error: {type(e).__name__}: {str(e)}"
