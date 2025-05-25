def calculate_aqi(concentration, pollutant):
    """
    Calculate Air Quality Index (AQI) based on pollutant concentration.
    
    Parameters:
    -----------
    concentration : float
        Pollutant concentration
    pollutant : str
        Pollutant type (PM2.5, PM10, etc.)
        
    Returns:
    --------
    float
        Calculated AQI value
    """
    # AQI breakpoints for different pollutants
    # Format: (Clow, Chigh, Ilow, Ihigh)
    aqi_breakpoints = {
        'PM2.5': [
            (0.0, 12.0, 0, 50),
            (12.1, 35.4, 51, 100),
            (35.5, 55.4, 101, 150),
            (55.5, 150.4, 151, 200),
            (150.5, 250.4, 201, 300),
            (250.5, 500.4, 301, 500),
        ],
        'PM10': [
            (0, 54, 0, 50),
            (55, 154, 51, 100),
            (155, 254, 101, 150),
            (255, 354, 151, 200),
            (355, 424, 201, 300),
            (425, 604, 301, 500),
        ],
        'CO': [
            (0.0, 4.4, 0, 50),
            (4.5, 9.4, 51, 100),
            (9.5, 12.4, 101, 150),
            (12.5, 15.4, 151, 200),
            (15.5, 30.4, 201, 300),
            (30.5, 50.4, 301, 500),
        ],
        'SO2': [
            (0, 35, 0, 50),
            (36, 75, 51, 100),
            (76, 185, 101, 150),
            (186, 304, 151, 200),
            (305, 604, 201, 300),
            (605, 1004, 301, 500),
        ],
        'NO2': [
            (0, 53, 0, 50),
            (54, 100, 51, 100),
            (101, 360, 101, 150),
            (361, 649, 151, 200),
            (650, 1249, 201, 300),
            (1250, 2049, 301, 500),
        ],
        'O3': [
            (0, 54, 0, 50),
            (55, 70, 51, 100),
            (71, 85, 101, 150),
            (86, 105, 151, 200),
            (106, 200, 201, 300),
            (201, 504, 301, 500),
        ]
    }
    
    # Default to PM2.5 if pollutant not found
    if pollutant not in aqi_breakpoints:
        pollutant = 'PM2.5'
    
    # Find the appropriate breakpoint
    for Clow, Chigh, Ilow, Ihigh in aqi_breakpoints[pollutant]:
        if Clow <= concentration <= Chigh:
            # Calculate AQI using formula
            aqi = ((Ihigh - Ilow) / (Chigh - Clow)) * (concentration - Clow) + Ilow
            return aqi
    
    # If concentration is higher than the highest breakpoint
    if concentration > aqi_breakpoints[pollutant][-1][1]:
        return 500  # Maximum AQI value
    
    # If concentration is lower than the lowest breakpoint
    return 0

def display_aqi_description(aqi):
    """
    Get the AQI category, color, and health message based on AQI value.
    
    Parameters:
    -----------
    aqi : float
        AQI value
        
    Returns:
    --------
    tuple
        (category, color, message)
    """
    if aqi <= 50:
        return ("Good", "#00e400", "Air quality is satisfactory, and air pollution poses little or no risk.")
    elif aqi <= 100:
        return ("Moderate", "#ffff00", "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution.")
    elif aqi <= 150:
        return ("Unhealthy for Sensitive Groups", "#ff7e00", "Members of sensitive groups may experience health effects. The general public is less likely to be affected.")
    elif aqi <= 200:
        return ("Unhealthy", "#ff0000", "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects.")
    elif aqi <= 300:
        return ("Very Unhealthy", "#99004c", "Health alert: The risk of health effects is increased for everyone.")
    else:
        return ("Hazardous", "#7e0023", "Health warning of emergency conditions: everyone is more likely to be affected.")
