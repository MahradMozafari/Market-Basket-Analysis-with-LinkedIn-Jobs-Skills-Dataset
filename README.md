# Market-Basket-Analysis-with-LinkedIn-Jobs-Skills-Dataset
Market-basket analysis is a popular technique used to identify relationships between items purchased together. In this project, we will apply this technique to the LinkedIn Jobs &amp; Skills dataset, focusing on the job skills listed in various job postings.
Project Title: Market-Basket Analysis on LinkedIn Jobs & Skills Dataset
Overview
This project performs market-basket analysis on the LinkedIn Jobs & Skills dataset. The goal is to find frequent itemsets (i.e., combinations of skills) that often appear together in job listings using Apriori and FP-Growth algorithms.

Files Included
job_skills.csv: The dataset containing job skills information.
market_basket_analysis.py: The Python script containing classes and methods for data preprocessing, market-basket analysis, and result interpretation.
Steps
Data Preprocessing:

Remove Missing Values: Handles missing values by either dropping records with missing job skills or filling them with empty strings.
Remove Duplicates: Removes duplicate records based on job skills.
Split Skills: Splits job skills into individual items.
Encode Skills: Encodes job skills into a basket structure suitable for market-basket analysis.
Market-Basket Analysis:

Frequent Itemsets: Uses Apriori and FP-Growth algorithms to find frequent itemsets.
Association Rules: Generates association rules based on frequent itemsets with specified metrics and thresholds.
Result Interpretation:

Top Rules: Displays the top association rules based on lift.
Visualization: Provides scatter plots of support vs confidence and heatmaps of association rules for better understanding.
Usage
Preprocessing Data:

python
Copy code
preprocessor = Preprocessor('job_skills.csv')
encoded_df = preprocessor.preprocess(missing_values_method='fillna', duplicate_subset='job_skills', skill_sep=',')
Market-Basket Analysis:

python
Copy code
analyzer = MarketBasketAnalyzer(encoded_df)

# Using Apriori algorithm
rules_apriori = analyzer.analyze(method='apriori', min_support=0.01, metric="lift", min_threshold=1)

# Using FP-Growth algorithm
rules_fpgrowth = analyzer.analyze(method='fpgrowth', min_support=0.01, metric="lift", min_threshold=1)
Result Interpretation:

python
Copy code
result_analyzer_apriori = ResultAnalyzer(rules_apriori)
result_analyzer_fpgrowth = ResultAnalyzer(rules_fpgrowth)

# Apriori Results
result_analyzer_apriori.show_top_rules(n=10)
result_analyzer_apriori.plot_support_vs_confidence()
result_analyzer_apriori.plot_heatmap()

# FP-Growth Results
result_analyzer_fpgrowth.show_top_rules(n=10)
result_analyzer_fpgrowth.plot_support_vs_confidence()
result_analyzer_fpgrowth.plot_heatmap()
Dependencies
pandas
matplotlib
seaborn
mlxtend
How to Run
Clone the repository.
Ensure you have the necessary dependencies installed.
Run the market_basket_analysis.py script.
bash
Copy code
python market_basket_analysis.py
This will preprocess the data, perform market-basket analysis, and display the results including visualizations.

