#Library importation and data loading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('all_data.csv')
df.info()
df.head(10)

#EDA Univariate Analysis of life expectancy
df['Life Expectancy'] = df['Life expectancy at birth (years)']
df['Life Expectancy'].describe()
plt.figure()
sns.histplot(df["Life Expectancy"], bins=15, kde=True)
plt.title("Distribution of Life Expectancy")
plt.xlabel("Life Expectancy (years)")
plt.ylabel("Frequency")
plt.show()

#EDA Univariate GDP
plt.figure()
sns.histplot(np.log(df['GDP']),bins = 15, kde = True)
plt.xlabel('GDP')
plt.ylabel('Frequency')
plt.title('Distribution of GDP')
plt.show()

#Life expectancy by country
sns.boxplot(data=df, x=df["Country"], y=df['Life Expectancy'])
plt.title("Life Expectancy by Country")
plt.xticks(rotation=45)
plt.show()

#GDP by country
sns.boxplot(data=df, x=df["Country"], y=np.log(df["GDP"]))
plt.title("GDP by Country")
plt.xticks(rotation=45)
plt.show()

#Bivariate analysis GDP vs Life Expectancy(LE)
plt.figure()
sns.scatterplot(df,x=np.log(df['GDP']),y=df['Life Expectancy'],hue=df['Country'])
plt.xlabel('GDP')
plt.ylabel('Life Expectancy')
plt.title('GDP vs Life Expectancy')
plt.show()

#Multivariate analysis LE overtime by country
plt.figure()
sns.lineplot(data=df, x="Year", y="life_expectancy", hue="country")
plt.title("Life Expectancy Over Time by Country")
plt.xlabel("Year")
plt.ylabel("Life Expectancy")
plt.show()

#Multivariate GDP overtime
plt.figure()
sns.lineplot(df, x= df['Year'], y=df['GDP'], hue=df['Country'])
plt.title('GDP Over Time by Country')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.show()

#GDP vs LE over time
sns.relplot(
    data=df,
    x=np.log(df['GDP']),
    y=df["Life Expectancy"],
    hue=df["Country"],
    size="Year",
    sizes=(20, 200),
    alpha=0.7
)
plt.title("GDP vs Life Expectancy Over Time")
plt.xlabel("Log GDP")
plt.ylabel("Life Expectancy")
plt.show()