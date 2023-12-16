def calculate_bmi(weight, height):
    # BMI Formula: BMI = weight (kg) / (height (m) * height (m))
    height_meters = height / 100  # Convert height from cm to meters
    bmi_temp = weight / (height_meters * height_meters)

    # Round the BMI result to two decimal places
    bmi = round(bmi_temp, 2)

    return bmi
