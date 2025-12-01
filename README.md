# UCBerkeley-Capstone-Assignment-Initial-Report-and-Exploratory-Data-Analysis-EDA
Exploratory data analysis, feature engineering, and baseline modeling for my capstone dataset.

# Predicting ADHD Patterns and Attention Variability Using Behavioral, Cognitive, and Lifestyle Data

## 1. Research Question

**Can behavioral, cognitive, and lifestyle data be used to predict attention variability and be used to design personalized interventions for individuals with ADHD?**

For Module 20.1, my focus is on:

- Cleaning and transforming the full feature set.
- Understanding the ADHD dataset through **exploratory data analysis (EDA)**.
- Building a **baseline multi-class classification model** to predict ADHD diagnosis class.
- Interpreting how behavioral, cognitive (questionnaire), and lifestyle variables relate to ADHD patterns.

---

## 2. Dataset

### 2.1 Source and Shape

- **Dataset Name in Kaggle:** ADHD Dataset – 4 Classes (u2)
- https://www.kaggle.com/datasets/a7md19/adhd-dataset-4-classes-u2/data
- **Rows:** ~6,500  
- **Columns:** 32 (31 features + 1 target)

The dataset is designed for mental and behavioral health analysis with a focus on ADHD diagnosis and subtypes. It combines demographics, lifestyle habits, and questionnaire-based symptom ratings.

### 2.2 Feature List and Meanings

#### Demographic and family history

- `Age`  
  Age of the participant (years).

- `Gender`  
  Encoded gender (e.g., 1 = male, 2 = female in the sample data; exact mapping from dataset documentation).

- `Educational_Level`  
  Education stage (e.g., Kindergarten, Primary, Middle, etc.).

- `Family_History`  
  Family history of ADHD or related mental/behavioral health issues (Yes/No).

#### Behavioral and lifestyle features

- `Sleep_Hours`  
  Typical daily sleep duration in hours.

- `Daily_Activity_Hours`  
  Time spent in physical or general activity per day (hours).

- `Daily_Phone_Usage_Hours`  
  Daily phone / screen usage time (hours).

- `Daily_Walking_Running_Hours`  
  Time spent walking or running per day (hours).

- `Daily_Coffee_Tea_Consumption`  
  Daily coffee/tea (caffeine) consumption, in cups or an ordinal scale.

#### Cognitive / functional / emotional indicators

- `Difficulty_Organizing_Tasks`  
  Degree of difficulty organizing tasks (ordinal score).

- `Focus_Score_Video`  
  Self-rated or observed focus while watching a video (higher = better focus).

- `Learning_Difficulties`  
  Presence or level of learning difficulties (ordinal score).

- `Anxiety_Depression_Levels`  
  Level of anxiety/depressive symptoms (ordinal score).

#### ADHD symptom questionnaires

The dataset included **18 questionnaire items** grouped into:

- **Q1\_1 – Q1\_9**: Hyperactivity / impulsivity items  
- **Q2\_1 – Q2\_9**: Inattention items  

Values are ordinal ratings (0–3) representing symptom severity or frequency in the sample data (e.g., 0 = not at all, 3 = very often).

These items closely correspond to the **nine DSM criteria for hyperactivity/impulsivity** and the **nine DSM criteria for inattention**, which are standard in ADHD assessment.   

**Hyperactivity / Impulsivity (Q1\_1 – Q1\_9)**

- `Q1_1`: Restlessness or fidgeting (e.g., often fidgets or taps hands/feet, squirms in seat).  
- `Q1_2`: Leaving seat when expected to remain seated.  
- `Q1_3`: Running or climbing in inappropriate situations / feeling constantly “on the go”.  
- `Q1_4`: Difficulty playing or engaging in leisure activities quietly.  
- `Q1_5`: Appearing “driven by a motor” / always on the move.  
- `Q1_6`: Talking excessively.  
- `Q1_7`: Blurting out answers or finishing other people’s sentences.  
- `Q1_8`: Difficulty waiting for their turn.  
- `Q1_9`: Interrupting or intruding on others’ activities or conversations.

**Inattention (Q2\_1 – Q2\_9)**

- `Q2_1`: Careless mistakes / not paying close attention to details.  
- `Q2_2`: Difficulty sustaining attention in tasks or play.  
- `Q2_3`: Not seeming to listen when spoken to directly.  
- `Q2_4`: Not following through on instructions / failing to finish tasks.  
- `Q2_5`: Difficulty organizing tasks and activities.  
- `Q2_6`: Avoiding or strongly disliking tasks requiring sustained mental effort.  
- `Q2_7`: Frequently losing things necessary for tasks (books, tools, etc.).  
- `Q2_8`: Easily distracted by external stimuli or unrelated thoughts.  
- `Q2_9`: Forgetfulness in daily activities (e.g., chores, errands).

These labels are used in this capstone for **interpretation and visualization**; the raw data columns themselves are numeric ratings.

#### Target variable

- `Diagnosis_Class` is a 4 class classification problem.
  - `0`: No ADHD  
  - `1`: Inattentive type  
  - `2`: Hyperactive-Impulsive type  
  - `3`: Combined type  

-

<img width="800" height="303" alt="pic1" src="https://github.com/user-attachments/assets/0057fa1d-47be-4193-b551-bf36cdff5163" />
---

## 3. Methods (Module 20.1)

### 3.1 Data Cleaning

Key steps implemented in the notebook:

- Loaded the dataset from Kaggle.
- Inspected data types, summary statistics, and class distribution.
- Check for:
  - Missing values (counts and percentage per column).
  - Duplicate records and remove exact duplicates.
- **Imputation:**
  - Numeric features: median imputation.
  - Categorical features (`Gender`, `Educational_Level`, `Family_History` if non-numeric): mode imputation.
- Confirm that no remaining missing values are used in modeling.

 


### 3.2 Feature Engineering

Feature engineering focuses on summarizing questionnaire items and creating a few behavior-based ratios:

- `hyperactivity_score`  
  Sum of `Q1_1`–`Q1_9` (overall hyperactivity/impulsivity severity).

- `inattention_score`  
  Sum of `Q2_1`–`Q2_9` (overall inattention severity).

- `activity_sleep_ratio`  
  `Daily_Activity_Hours` / (`Sleep_Hours` + small constant to avoid division by zero).  
  Captures balance between activity and sleep.
<img width="800" height="323" alt="pic3" src="https://github.com/user-attachments/assets/aaee6d73-6b59-46a5-a0b0-2ef4fa5223da" />

- `phone_usage_intensity`  
  `Daily_Phone_Usage_Hours` / 24.  
  Normalizes phone use relative to the full day.

The **full feature set** used in modeling includes:

- All original features  
  `Age`, `Gender`, `Educational_Level`, `Family_History`,  
  `Sleep_Hours`, `Daily_Activity_Hours`,  
  `Q1_1`–`Q1_9`, `Q2_1`–`Q2_9`,  
  `Daily_Phone_Usage_Hours`, `Daily_Walking_Running_Hours`,  
  `Difficulty_Organizing_Tasks`, `Focus_Score_Video`,  
  `Daily_Coffee_Tea_Consumption`, `Learning_Difficulties`,  
  `Anxiety_Depression_Levels`

**Plus** engineered features:  
`hyperactivity_score`, `inattention_score`, `activity_sleep_ratio`, `phone_usage_intensity`.

### 3.3 Exploratory Data Analysis (EDA)

- **Univariate distributions**
  - Histograms and KDE plots for continuous variables:
    - Sleep, phone usage, activity, walking/running, coffee/tea consumption.
    - Hyperactivity and inattention total scores.
    - Focus score and anxiety/depression levels.
<img width="1024" height="503" alt="pic4" src="https://github.com/user-attachments/assets/25a8808c-6575-4c26-af51-ab43cf18a309" />

- **Target distribution**
  - Count plot of `Diagnosis_Class` to understand class balance across the 4 categories.
<img width="640" height="378" alt="pic2" src="https://github.com/user-attachments/assets/e012b71e-cf8e-45aa-a298-09ac3f13402b" />

- **Bivariate relationships**
  - Boxplots of key behavioral and questionnaire features across `Diagnosis_Class`.
  - Group-level means/medians of:
    - `hyperactivity_score`
    - `inattention_score`
    - `Sleep_Hours`
    - `Daily_Phone_Usage_Hours`
    - `Anxiety_Depression_Levels`

- **Correlation structure**
  - Correlation matrix and heatmap for:
    - Behavioral features
    - Questionnaire scores
    - Emotional indicators (`Anxiety_Depression_Levels`)
  - Focus on how hyperactivity, inattention, and lifestyle factors relate.
<img width="800" height="636" alt="pic5" src="https://github.com/user-attachments/assets/18f6552f-f464-481f-9623-5b94a83c4a57" />

- **Outlier analysis**
  - Boxplots for variables such as `Sleep_Hours`, `Daily_Phone_Usage_Hours`, `hyperactivity_score`, `inattention_score`.
  - Qualitative decision about whether to keep or cap extreme values.

### 3.4 Baseline Model

#### Attempt 1:
I started with  **LogisticRegression multi-class classification model** i:

- **Model:** Multinomial Logistic Regression 
- **Inputs:**  
  - All numeric-encoded versions of:
    - Demographic / history.
    - Behavioral and lifestyle variables.
    - All 18 questionnaire items (Q1\_1–Q1\_9, Q2\_1–Q2\_9).
    - Cognitive and emotional indicators.
    - Engineered scores and ratios.

- **Train/test split:** 80% train, 20% test, stratified by `Diagnosis_Class`.

- **Scaling:**  
  - Standardization with `StandardScaler` applied to features for logistic regression.

- **Primary evaluation metric:**  
  - **Macro F1-score** (averages F1 across all four classes, treating each class equally).

 <img width="640" height="334" alt="pic6" src="https://github.com/user-attachments/assets/cdb239c7-d1fa-4fc0-af6e-f5b151b8d555" />

**Metric analysis **

FI-score of 1 means there is data leakage.

#### Attempt 2:
Next Itrained the model only with:
Demographics, lifestyle, and other indicators (Age, Gender, Educational_Level, Family_History, Sleep_Hours, Daily_Activity_Hours, Daily_Phone_Usage_Hours, Daily_Walking_Running_Hours, Daily_Coffee_Tea_Consumption, Difficulty_Organizing_Tasks, Focus_Score_Video, Learning_Difficulties, Anxiety_Depression_Levels). I dropped Q1_1–Q1_9, Q2_1–Q2_9, Hyperactivity_Score, Inattention_Score as these features directly encode the label and may be the reason for leakage and so the model is learning nothing.

<img width="640" height="317" alt="pic7" src="https://github.com/user-attachments/assets/f9642a96-4987-461a-9ee4-759b7c6d711b" />

class 1 recall is 0.25 so many class 1 is missclassified.
Class 2 F1-score precision and recall are 0 , so model is not predicting class 2
This could be due to class imbalance. I will try again using class_weight="balanced"
---
#### Attempt 3:
Next I trained the model everthing above and class_weight="balanced":

<img width="640" height="368" alt="pic8" src="https://github.com/user-attachments/assets/7ae38926-7a42-4ad2-b5e4-d86466179ca6" />

Accuracy ≈ 0.50
Macro F1 ≈ 0.48
Non-zero performance for all 4 classes, including 1 and 2.

This improved the balance and F1 score though accuracy went down. 

<img width="640" height="516" alt="pic9" src="https://github.com/user-attachments/assets/0a9e90c5-6ac6-45ee-a82e-475e3a0bec2d" />

I will make the multinomial logistic regression with balanced class weights using non-questionnaire features only as the baseline model for my capstone project.


## 4. Summary

Summary
Composite scores: I first turned the 18 symptom questions into two simple scores:

Hyperactivity_Score = Q1_1–Q1_9 added together (range 0–27)

Inattention_Score = Q2_1–Q2_9 added together (range 0–27)

Performed EDA

Looked at how the four Diagnosis_Class labels are distributed.

Plotted basic distributions for sleep, phone use, movement, mood, and the two new symptom scores.

Used boxplots of Hyperactivity_Score and Inattention_Score by Diagnosis_Class. These showed almost perfect separation between classes, which makes it clear that the Q1/Q2 items are essentially encoding the diagnosis rule.

Checked correlations between features and scanned for outliers in the main continuous variables.

This revealed label leakage:

To avoid this, I built a baseline multinomial logistic regression model using only non-symptom features:

Inputs limited to demographics, lifestyle and other indicators (Age, Gender, Educational_Level, Family_History, Sleep_Hours, Daily_Activity_Hours, Daily_Phone_Usage_Hours, Daily_Walking_Running_Hours, Daily_Coffee_Tea_Consumption, Difficulty_Organizing_Tasks, Focus_Score_Video, Learning_Difficulties, Anxiety_Depression_Levels).

Categorical variables were one-hot encoded and all features were standardized.

Used class_weight="balanced" so minority classes are not ignored.

Model results:

Test accuracy around 0.50, macro F1 around 0.48.

All four classes (0–3) have non-zero F1, with better recall and F1 for classes 1 and 2 than the unweighted version.

This gives a realistic, clinically sensible baseline that I can now use to compare more complex models and resampling approaches later.
