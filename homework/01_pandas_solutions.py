'''
SOLUTIONS: Working with drinks data
'''
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

# Read drinks.csv into a DataFrame called 'drinks'
drinks = pd.read_csv('../data/drinks.csv')

# Print the first 10 rows
drinks.head(10)

# Examine the data types of all columns
drinks.dtypes
drinks.info()

# Print the 'beer_servings' Series
drinks.beer_servings
drinks['beer_servings']

# Calculate the average 'beer_servings' for the entire dataset
drinks.describe()                   # summarize all numeric columns
drinks.beer_servings.describe()     # summarize only the 'beer_servings' Series
drinks.beer_servings.mean()         # only calculate the mean

# Print all columns, but only show rows where the country is in Europe
drinks[drinks.continent=='EU']

# Calculate the average 'beer_servings' for all of Europe
drinks[drinks.continent=='EU'].beer_servings.mean()

# Only show European countries with 'wine_servings' greater than 300
drinks[(drinks.continent=='EU') & (drinks.wine_servings > 300)]

# Determine which 10 countries have the highest 'total_litres_of_pure_alcohol'
drinks.sort_index(by='total_litres_of_pure_alcohol').tail(10)

# Determine which country has the highest value for 'beer_servings'
drinks[drinks.beer_servings==drinks.beer_servings.max()].country
drinks[['country', 'beer_servings']].sort_index(by='beer_servings', ascending=False).head(5)

# Count the number of occurrences of each 'continent' value and see if it looks correct
drinks.continent.value_counts()

# Determine which countries do not have continent designations
drinks[drinks.continent.isnull()].country


'''
Plotting
'''

# bar plot of number of countries in each continent
drinks.continent.value_counts().plot(kind='bar', title='Countries per Continent')
plt.xlabel('Continent')
plt.ylabel('Count')
plt.show()                                  # show plot window (if it doesn't automatically appear)
plt.savefig('countries_per_continent.png')  # save plot to file

# bar plot of average number of beer servings (per adult per year) by continent
drinks.groupby('continent').beer_servings.mean().plot(kind='bar')
plt.ylabel('Average Number of Beer Servings Per Year')

# histogram of beer servings (shows the distribution of a numeric column)
drinks.beer_servings.hist(bins=20)
plt.xlabel('Beer Servings')
plt.ylabel('Frequency')

# density plot of beer servings (smooth version of a histogram)
drinks.beer_servings.plot(kind='density', xlim=(0,500))
plt.xlabel('Beer Servings')

# grouped histogram of beer servings (shows the distribution for each group)
drinks.beer_servings.hist(by=drinks.continent)
drinks.beer_servings.hist(by=drinks.continent, sharex=True)
drinks.beer_servings.hist(by=drinks.continent, sharex=True, sharey=True)
drinks.beer_servings.hist(by=drinks.continent, layout=(1, 5))   # change layout (new in pandas 0.15.0)

# boxplot of beer servings by continent (shows five-number summary and outliers)
drinks.boxplot(column='beer_servings', by='continent')

# scatterplot of beer servings versus wine servings
drinks.plot(kind='scatter', x='beer_servings', y='wine_servings', alpha=0.3)

# same scatterplot, except point color varies by 'spirit_servings'
# note: must use 'c=drinks.spirit_servings' prior to pandas 0.15.0
drinks.plot(kind='scatter', x='beer_servings', y='wine_servings', c=drinks.spirit_servings, colormap='Blues')

# same scatterplot, except all European countries are colored red
colors = np.where(drinks.continent=='EU', 'r', 'b')
drinks.plot(x='beer_servings', y='wine_servings', kind='scatter', c=colors)

# scatterplot matrix of all numerical columns
pd.scatter_matrix(drinks)