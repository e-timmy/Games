from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

driver = webdriver.Chrome() # Or Firefox(), etc.
driver.get("www.google.com") # Replace with the actual URL

wait = WebDriverWait(driver, 10)

try:
    # Locate the input field
    date_input = wait.until(EC.presence_of_element_located((By.ID, "date-input-id"))) # Replace with actual ID/locator

    # Clear existing value (optional but often good practice)
    date_input.clear()

    # Send the date string - Make sure the format matches what the website expects!
    date_input.send_keys("MM/DD/YYYY") # e.g., "09/25/2024"

    # Sometimes you need to press Enter or Tab to confirm
    # from selenium.webdriver.common.keys import Keys
    # date_input.send_keys(Keys.RETURN)

    time.sleep(2) # Pause to see the result

except Exception as e:
    print(f"Error interacting with date input: {e}")

finally:
    driver.quit()