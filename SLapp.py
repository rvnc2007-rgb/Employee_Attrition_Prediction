import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the exported model and transformer
@st.cache_resource
def load_assets():
    model = joblib.load('logistic_regression_model.pkl')
    transformer = joblib.load('column_transformer.pkl')
    return model, transformer

model, transformer = load_assets()

# 2. App Header Text
st.title("📊 HR Employee Attrition Predictor")
st.markdown("Adjust employee details below to see real-time attrition risks and personalized drivers.")

st.divider()

# 3. Create the UI Input Sections
st.subheader("📋 Employee Profile Data")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 65, 35)
    monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=5000)
    distance_from_home = st.slider("Distance From Home (miles)", 1, 30, 5)
    total_working_years = st.number_input("Total Working Years", 0, 40, 10)
    years_at_company = st.number_input("Years At Company", 0, 40, 5)
    years_in_current_role = st.number_input("Years In Current Role", 0, 20, 2)
    years_since_last_promotion = st.number_input("Years Since Last Promotion", 0, 15, 1)
    years_with_curr_manager = st.number_input("Years With Current Manager", 0, 20, 2)
    num_companies_worked = st.number_input("Number of Companies Worked", 0, 10, 2)
    percent_salary_hike = st.slider("Percent Salary Hike (%)", 0, 25, 12)
    training_times_last_year = st.slider("Training Times Last Year", 0, 6, 2)
    
    # Static data variables matching your dataset columns
    daily_rate = 800 
    hourly_rate = 65
    monthly_rate = 14000

with col2:
    business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
    department = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
    education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    job_role = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    overtime = st.selectbox("Overtime Worked?", ["Yes", "No"])
    over_18 = "Y" # Static rule

    # Categorical/Numerical placeholders for columns left un-transformed by your pipeline
    environment_satisfaction = 3
    education = 3
    job_involvement = 3
    job_level = 2
    job_satisfaction = 3
    performance_rating = 3
    relationship_satisfaction = 3
    work_life_balance = 3
    stock_option_level = 1

st.divider()

# 4. Process and Predict
if st.button("🔮 Predict Attrition Risk", type="primary"):
    # Construct a dictionary matching original feature structure perfectly
    input_data = {
        'Age': age, 'BusinessTravel': business_travel, 'DailyRate': daily_rate,
        'Department': department, 'DistanceFromHome': distance_from_home, 'Education': education,
        'EducationField': education_field, 'EnvironmentSatisfaction': environment_satisfaction, 'Gender': gender,
        'HourlyRate': hourly_rate, 'JobInvolvement': job_involvement, 'JobLevel': job_level,
        'JobRole': job_role, 'JobSatisfaction': job_satisfaction, 'MaritalStatus': marital_status,
        'MonthlyIncome': monthly_income, 'MonthlyRate': monthly_rate, 'NumCompaniesWorked': num_companies_worked,
        'Over18': over_18, 'OverTime': overtime, 'PercentSalaryHike': percent_salary_hike,
        'PerformanceRating': performance_rating, 'RelationshipSatisfaction': relationship_satisfaction,
        'StockOptionLevel': stock_option_level, 'TotalWorkingYears': total_working_years,
        'TrainingTimesLastYear': training_times_last_year, 'WorkLifeBalance': work_life_balance,
        'YearsAtCompany': years_at_company, 'YearsInCurrentRole': years_in_current_role,
        'YearsSinceLastPromotion': years_since_last_promotion, 'YearsWithCurrManager': years_with_curr_manager
    }
    
    # Convert into a 1-row DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Apply your preprocessing column transformer
    transformed_input = transformer.transform(input_df)
    
    # Get model calculations
    prediction = model.predict(transformed_input)
    probability = model.predict_proba(transformed_input)[0][1] # Probability of "Yes"
    
    # Display the metrics cleanly
    st.subheader("📊 Model Prediction Results")
    
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        if prediction == 'Yes' or probability > 0.5:
            st.error(f"⚠️ **High Attrition Risk:** This employee is statistically likely to leave.")
        else:
            st.success(f"✅ **Low Attrition Risk:** This employee is likely to stay.")
            
    with res_col2:
        st.metric(label="Calculated Attrition Probability", value=f"{probability * 100:.2f}%")
        
    st.divider()
    
    # 5. REAL-TIME FACTOR CALCULATION
    st.subheader("🔍 Personal Attrition Drivers (Real-Time)")
    st.markdown("This chart breaks down the features specifically pushing **this individual** out the door vs. keeping them here right now.")
    
    try:
        # Get feature names and weights
        feature_names = transformer.get_feature_names_out()
        coefficients = model.coef_[0]
        
        # Calculate real-time contribution (Preprocessed Value * Coefficient Weight)
        employee_values = transformed_input[0]
        real_time_contribution = employee_values * coefficients
        
        # Build DataFrame
        real_time_df = pd.DataFrame({
            'Feature': feature_names,
            'Contribution': real_time_contribution
        })
        
        # Clean feature names for clean text output
        real_time_df['Feature'] = real_time_df['Feature'].str.replace('num__', '').str.replace('cat__', '')
        
        # Filter out features that have 0 impact for this specific user (e.g., non-active category checkboxes)
        real_time_df = real_time_df[real_time_df['Contribution'] != 0]
        
        # Sort values to see top risk factors (positive scores mean pushing towards leaving)
        real_time_df['Abs_Contribution'] = real_time_df['Contribution'].abs()
        top_drivers = real_time_df.sort_values(by='Abs_Contribution', ascending=False).head(8)
        top_drivers = top_drivers.sort_values(by='Contribution', ascending=True)
        
        # Plot using Matplotlib & Seaborn
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#2b5c8f' if x < 0 else '#d9534f' for x in top_drivers['Contribution']]
        
        sns.barplot(
            x='Contribution', 
            y='Feature', 
            data=top_drivers, 
            palette=colors,
            hue='Feature',
            legend=False,
            ax=ax
        )
        
        ax.set_title("Top Real-Time Impact Factors For This Employee", fontsize=14, pad=15)
        ax.set_xlabel("Impact Score (Right = Pushing to Leave | Left = Encouraging to Stay)", fontsize=11)
        ax.set_ylabel("")
        plt.tight_layout()
        
        # Render plot
        st.pyplot(fig)
        
        # Actionable written takeaway for HR
        st.markdown("### 💡 Strategic Retention Insights:")
        top_negative_factor = top_drivers.sort_values(by='Contribution', ascending=False).iloc[0]
        
        if top_negative_factor['Contribution'] > 0:
            st.warning(f"The number one real-time factor pushing this employee to leave is **{top_negative_factor['Feature']}**. Addressing this specific condition is your best lever for retaining them.")
        else:
            st.info("This employee currently has strong retention anchors keeping them happy at the company.")
            
    except Exception as e:
        st.warning(f"Could not automatically render the real-time importance chart. Error: {e}")
