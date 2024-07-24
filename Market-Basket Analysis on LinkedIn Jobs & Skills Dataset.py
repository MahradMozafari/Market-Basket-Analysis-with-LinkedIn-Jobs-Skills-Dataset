import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

class Preprocessor:
    def __init__(self, filepath):
        # Load data from CSV file
        self.df = pd.read_csv(filepath)
        
    def remove_missing_values(self, method='drop'):
        # Remove or fill missing values
        if method == 'drop':
            self.df = self.df.dropna(subset=['job_skills'])
        elif method == 'fillna':
            self.df['job_skills'] = self.df['job_skills'].fillna('')
        return self.df

    def remove_duplicates(self, subset='job_skills'):
        # Remove duplicate records
        self.df = self.df.drop_duplicates(subset=[subset])
        return self.df

    def split_skills(self, sep=','):
        # Split job skills into individual items
        self.df['job_skills'] = self.df['job_skills'].str.split(sep)
        return self.df

    def encode_skills(self):
        # Encode job skills into a basket structure
        all_skills = set(skill.strip() for sublist in self.df['job_skills'] for skill in sublist)
        encoded_vals = []
        for _, row in self.df.iterrows():
            rowset = set(skill.strip() for skill in row['job_skills'])
            encoded_vals.append({skill: (skill in rowset) for skill in all_skills})
        encoded_df = pd.DataFrame(encoded_vals)
        return encoded_df

    def preprocess(self, missing_values_method='drop', duplicate_subset='job_skills', skill_sep=','):
        # Complete preprocessing pipeline
        self.remove_missing_values(method=missing_values_method)
        self.remove_duplicates(subset=duplicate_subset)
        self.split_skills(sep=skill_sep)
        encoded_df = self.encode_skills()
        return encoded_df


class MarketBasketAnalyzer:
    def __init__(self, encoded_df):
        # Initialize with preprocessed data
        self.encoded_df = encoded_df
        self.frequent_itemsets = None
        self.rules = None

    def find_frequent_itemsets_apriori(self, min_support=0.01):
        # Find frequent itemsets using Apriori
        self.frequent_itemsets = apriori(self.encoded_df, min_support=min_support, use_colnames=True)
        return self.frequent_itemsets

    def find_frequent_itemsets_fpgrowth(self, min_support=0.01):
        # Find frequent itemsets using FP-Growth
        self.frequent_itemsets = fpgrowth(self.encoded_df, min_support=min_support, use_colnames=True)
        return self.frequent_itemsets

    def generate_association_rules(self, metric="lift", min_threshold=1):
        # Generate association rules
        if self.frequent_itemsets is not None:
            self.rules = association_rules(self.frequent_itemsets, metric=metric, min_threshold=min_threshold)
            return self.rules
        else:
            raise ValueError("Frequent itemsets not found. Please run an itemset finding method first.")

    def analyze(self, method='apriori', min_support=0.01, metric="lift", min_threshold=1):
        # Perform market basket analysis
        if method == 'apriori':
            self.find_frequent_itemsets_apriori(min_support)
        elif method == 'fpgrowth':
            self.find_frequent_itemsets_fpgrowth(min_support)
        else:
            raise ValueError("Unsupported method. Choose 'apriori' or 'fpgrowth'.")

        self.generate_association_rules(metric, min_threshold)
        return self.rules


class ResultAnalyzer:
    def __init__(self, rules):
        # Initialize with association rules
        self.rules = rules

    def show_top_rules(self, n=10):
        # Show top n association rules
        top_rules = self.rules.sort_values(by='lift', ascending=False).head(n)
        print("Top Association Rules:")
        print(top_rules)
        return top_rules

    def plot_support_vs_confidence(self):
        # Plot support vs confidence scatter plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x="support", y="confidence", size="lift", data=self.rules, legend=False)
        plt.title('Support vs Confidence')
        plt.xlabel('Support')
        plt.ylabel('Confidence')
        plt.show()

    def plot_heatmap(self):
        # Plot heatmap of association rules
        plt.figure(figsize=(12, 8))
        rules_pivot = self.rules.pivot("antecedents", "consequents", "lift")
        sns.heatmap(rules_pivot, annot=True, cmap="YlGnBu", cbar=True)
        plt.title('Heatmap of Association Rules')
        plt.show()


# Create Preprocessor object and preprocess data
preprocessor = Preprocessor('job_skills.csv')
encoded_df = preprocessor.preprocess(missing_values_method='fillna', duplicate_subset='job_skills', skill_sep=',')

# Create MarketBasketAnalyzer object
analyzer = MarketBasketAnalyzer(encoded_df)

# Market basket analysis using Apriori algorithm
rules_apriori = analyzer.analyze(method='apriori', min_support=0.01, metric="lift", min_threshold=1)
print("Association Rules using Apriori:")
print(rules_apriori)

# Market basket analysis using FP-Growth algorithm
rules_fpgrowth = analyzer.analyze(method='fpgrowth', min_support=0.01, metric="lift", min_threshold=1)
print("Association Rules using FP-Growth:")
print(rules_fpgrowth)

# Create ResultAnalyzer object for result interpretation
result_analyzer_apriori = ResultAnalyzer(rules_apriori)
result_analyzer_fpgrowth = ResultAnalyzer(rules_fpgrowth)

# Display and interpret top rules using Apriori algorithm
top_rules_apriori = result_analyzer_apriori.show_top_rules(n=10)
result_analyzer_apriori.plot_support_vs_confidence()
result_analyzer_apriori.plot_heatmap()

# Display and interpret top rules using FP-Growth algorithm
top_rules_fpgrowth = result_analyzer_fpgrowth.show_top_rules(n=10)
result_analyzer_fpgrowth.plot_support_vs_confidence()
result_analyzer_fpgrowth.plot_heatmap()
