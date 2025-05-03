import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import numpy as np

# Set seaborn style for better visualization
sns.set_style("whitegrid")

def load_and_explore_data():
    """Load and explore the Iris dataset"""
    try:
        # Load iris dataset from sklearn
        iris = load_iris()
        # Create DataFrame
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
        
        print("First few rows of the dataset:")
        print(df.head())
        print("\nDataset Info:")
        print(df.info())
        print("\nMissing Values:")
        print(df.isnull().sum())
        
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def analyze_data(df):
    """Perform basic data analysis"""
    try:
        print("\nBasic Statistics:")
        print(df.describe())
        
        print("\nMean measurements by species:")
        group_means = df.groupby('species').mean()
        print(group_means)
        
        # Additional analysis: correlation between features
        print("\nCorrelation Matrix:")
        print(df.iloc[:, :-1].corr())
        
        return group_means
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        return None

def create_visualizations(df, group_means):
    """Create required visualizations"""
    try:
        # Create figure with subplots
        plt.figure(figsize=(15, 10))
        
        # 1. Line chart (using sepal length as pseudo-time series)
        plt.subplot(2, 2, 1)
        for species in df['species'].unique():
            species_data = df[df['species'] == species]
            plt.plot(species_data.index, species_data['sepal length (cm)'], 
                    label=species, marker='o')
        plt.title('Sepal Length Trends by Species')
        plt.xlabel('Sample Index')
        plt.ylabel('Sepal Length (cm)')
        plt.legend()
        
        # 2. Bar chart (mean sepal length by species)
        plt.subplot(2, 2, 2)
        group_means['sepal length (cm)'].plot(kind='bar')
        plt.title('Mean Sepal Length by Species')
        plt.xlabel('Species')
        plt.ylabel('Mean Sepal Length (cm)')
        plt.xticks(rotation=45)
        
        # 3. Histogram (petal length distribution)
        plt.subplot(2, 2, 3)
        sns.histplot(data=df, x='petal length (cm)', hue='species', 
                    multiple='stack')
        plt.title('Petal Length Distribution')
        plt.xlabel('Petal Length (cm)')
        plt.ylabel('Count')
        
        # 4. Scatter plot (sepal length vs petal length)
        plt.subplot(2, 2, 4)
        sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', 
                       hue='species', size='species')
        plt.title('Sepal Length vs Petal Length')
        plt.xlabel('Sepal Length (cm)')
        plt.ylabel('Petal Length (cm)')
        
        plt.tight_layout()
        plt.savefig('iris_visualizations.png')
        plt.close()
        
        print("\nVisualizations have been saved as 'iris_visualizations.png'")
    except Exception as e:
        print(f"Error in visualization: {str(e)}")

def main():
    print("Iris Dataset Analysis")
    print("=" * 50)
    
    # Load and explore data
    df = load_and_explore_data()
    if df is None:
        return
    
    # Analyze data
    group_means = analyze_data(df)
    if group_means is None:
        return
    
    # Create visualizations
    create_visualizations(df, group_means)
    
    print("\nKey Findings:")
    print("- Setosa species has the smallest average measurements across all features")
    print("- Virginica species generally has the largest measurements")
    print("- Petal measurements show clearer separation between species than sepal measurements")
    print("- Strong positive correlation between petal length and petal width")

if __name__ == "__main__":
    main()