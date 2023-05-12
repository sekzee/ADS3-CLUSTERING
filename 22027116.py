# -*- coding: utf-8 -*-
"""
Created on Fri May 12 09:26:10 2023

@author: User
"""
import pandas as pd
import numpy as np
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import matplotlib.pyplot as plt
import cluster_tools as ct
import scipy.optimize as opt
import errors as err
from scipy.stats import t
import matplotlib.patches as mpatches
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D
from numpy.polynomial.polynomial import Polynomial

df_Gdp_Agric = pd.read_csv("GDPofAgric.csv", skiprows=4)
df_Gdp_Agric = df_Gdp_Agric.dropna().drop(['Indicator Code', 'Country Code', 'Indicator Name'], axis=1)
df_Gdp_Agric = df_Gdp_Agric.set_index('Country Name')

print(df_Gdp_Agric)

# Select 80 random countries
random_countries = df_Gdp_Agric.sample(n=50, replace=True, random_state=42)
# Print the selected countries
print(random_countries)

# List of countries to remove not needed
countries_to_remove = ['East Asia & Pacific (excluding high income)',
                       'East Asia & Pacific',
                       'IBRD only',
                       'Upper middle income',
                       'Low & middle income',
                       'South Asia (IDA & IBRD)',
                       'Late-demographic dividend',
                       'Iran, Islamic Rep.',
                       'East Asia & Pacific (IDA & IBRD countries)',
                       'IDA & IBRD total',
                       'IDA & IBRD total','Middle income','South Asia']

# Remove the countries from the DataFrame
selected_countries = df_Gdp_Agric[~df_Gdp_Agric.index.isin(countries_to_remove)]
# Print the resulting DataFrame

print(selected_countries)

# Print summary statistics of the DataFrame
print(selected_countries)

# Select the desired years for agricultural data
years = ["1970", "1980", "1990", "2000", "2010", "2015"]

# Create a new DataFrame with selected years of agricultural data
df_Agric_countries = selected_countries[years]

# Display the resulting DataFrame
print(df_Agric_countries)

# Plot scatter matrix
pd.plotting.scatter_matrix(df_Agric_countries, figsize=(12, 12), s=5, alpha=0.8)

# Display the plot
plt.show()

# extract columns for fitting. 
# .copy() prevents changes in df_fit to affect df_fish.
df_cluster = df_Agric_countries[["1970", "2000"]].copy()

# normalise dataframe and inspect result
# normalisation is done only on the extract columns. .copy() prevents
# changes in df_cluster to affect df_Gdp_Agric. This make the plots with the 
# original measurements
df_cluster, df_min, df_max = ct.scaler(df_cluster)
print
#print(df_cluster.describe())
#print()

print("n   score")
# loop over trial numbers of clusters calculating the silhouette
for ic in range(2, 7):
    # set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(df_cluster)     

    # extract labels and calculate silhoutte score
    labels = kmeans.labels_
    print (ic, skmet.silhouette_score(df_cluster, labels))
# Plot for four clusters
nc = 4 # number of cluster centres

kmeans = cluster.KMeans(n_clusters=nc)
kmeans.fit(df_cluster)     

# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_


# Add cluster label column to the original dataframe
df_Agric_countries["cluster_label"] = kmeans.labels_

# Group countries by cluster label
grouped = df_Agric_countries.groupby("cluster_label")

# Print countries in each cluster
for label, group in grouped:
    print("Cluster", label)
    print(group.index.tolist())
    print()

# Plot
plt.figure(figsize=(6.0, 6.0))
scatter = plt.scatter(df_cluster["1970"], df_cluster["2000"], c=labels, cmap='viridis')
xc = cen[:,0]
yc = cen[:,1]
plt.scatter(xc, yc, c="k", marker="d", s=80)

# Set labels
plt.xlabel("1970")
plt.ylabel("2000")
plt.title("Agriculture, forestry, and fishing, value added (% of GDP)")

# Define cluster colors and labels
colors = ['purple', 'green', 'blue', 'yellow']
cluster_labels = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3']

# Create custom legend
custom_legend = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=5) for c in colors]
plt.legend(custom_legend, cluster_labels, title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()

clusters = {
    0: ['Botswana', 'China', "Cote d'Ivoire", 'Costa Rica', 'Ecuador', 'Egypt, Arab Rep.', 'Honduras', 'Korea, Rep.', 'Lesotho', 'Malaysia', 'Philippines', 'Senegal', 'Eswatini', 'Thailand', 'Turkiye'],
    1: ['Benin', 'Burkina Faso', 'Guyana', 'India', 'Kenya', 'Sri Lanka', 'Pakistan', 'Togo'],
    2: ['Brazil', 'Chile', 'Congo, Rep.', 'France', 'Gabon', 'Singapore', 'Suriname', 'South Africa', 'Zambia'],
    3: ['Bangladesh', 'Ghana', 'Malawi', 'Niger', 'Sudan', 'Chad', 'Uganda']
}

print(clusters)

# Extract countries for each cluster
cluster_countries = {}
for label, group in grouped:
    cluster_countries[label] = group.index.tolist()

print(cluster_countries)

# Read file with population data into DataFrame
df_popG = pd.read_csv("POPgrowth.csv", skiprows = 4)


df_popG = df_popG.dropna(how='all')
df_popG = df_popG.drop('1960', axis=1)

df_popG = df_popG.drop(['Indicator Code', 'Country Code', 'Indicator Name', 'Unnamed: 66'], axis=1)
df_popG = df_popG.set_index('Country Name')

print(df_popG)

countries = ['India', 'China','Uganda','France']

df_popG_countries = df_popG.loc[countries]
df_popG_countries = df_popG_countries.transpose()
df_popG_countries = df_popG_countries.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
df_popG_countries = df_popG_countries.rename_axis('Year')

print(df_popG_countries)

fig, axs = plt.subplots(2, 2, figsize=(12, 12))  # Create a 2x2 grid of subplots

# Iterate over each country and each subplot
for ax, country in zip(axs.flatten(), ['India', 'China','Uganda','France']):
    # Get the data for the current country
    x_data = df_popG_countries.index.values.astype(int)
    y_data = df_popG_countries[country].values

    # Fit the polynomial (degree 3 in this case) to the data
    p = Polynomial.fit(x_data, y_data, 3)

    # Generate x values for the fitted function
    x_fit = np.linspace(x_data.min(), x_data.max(), 1000)

    # Calculate the fitted y values
    y_fit = p(x_fit)

    # Plot the original data and the fitted function
    ax.plot(x_data, y_data, 'bo')
    ax.plot(x_fit, y_fit, 'r-')
    ax.set_title(country)
    ax.set_xlabel('Year')
    ax.set_ylabel('Population')

plt.tight_layout()
plt.show()

def poly(t, c0, c1, c2, c3):
    """ Computes a polynomial c0 + c1*t + c2*t^2 + c3*t^3 """
    t = t - 1950
    f = c0 + c1*t + c2*t**2 + c3*t**3
    return f

# Ensure index is of integer type
df_popG_countries.index = df_popG_countries.index.astype(int)

popt, pcorr = opt.curve_fit(poly, df_popG_countries.index, df_popG_countries["France"])
print("Fit parameter", popt)
# extract variances and calculate sigmas
sigmas = np.sqrt(np.diag(pcorr))
# call function to calculate upper and lower limits with extrapolation
# create extended year range
years = np.arange(1950, 2051)
lower, upper = err.err_ranges(years, poly, popt, sigmas)
df_popG_countries["poly"] = poly(df_popG_countries.index, *popt)

plt.figure()
plt.title("Polynomial Fit")
plt.plot(df_popG_countries.index, df_popG_countries["France"], label="data")
plt.plot(df_popG_countries.index, df_popG_countries["poly"], label="fit")
# plot error ranges with transparency
plt.fill_between(years, lower, upper, alpha=0.5)
plt.legend(loc="upper left")
plt.show()

# Define the model function
def poly(t, c0, c1, c2, c3):
    """ Computes a polynomial c0 + c1*t + c2*t^2 + c3*t^3 """
    t = t - 1950
    f = c0 + c1*t + c2*t**2 + c3*t**3
    return f

def err_ranges(x, func, popt, perr):
    """ Calculate upper and lower errors """
    popt_up = popt + perr
    popt_dw = popt - perr
    fit = func(x, *popt)
    fit_up = func(x, *popt_up)
    fit_dw = func(x, *popt_dw)
    return fit_up, fit_dw

# Ensure index is of integer type
df_popG_countries.index = df_popG_countries.index.astype(int)

# Initialize a figure
fig, axs = plt.subplots(2, 2, figsize=(10,10))

# Flattening axs for easy iterating
axs = axs.ravel()

# Loop over the countries list
for i, country in enumerate(countries):
    popt, pcorr = curve_fit(poly, df_popG_countries.index, df_popG_countries[country])
    print(f"Fit parameters for {country}: ", popt)
    # extract variances and calculate sigmas
    sigmas = np.sqrt(np.diag(pcorr))
    # call function to calculate upper and lower limits with extrapolation
    # create extended year range
    years = np.arange(1950, 2051)
    lower, upper = err_ranges(years, poly, popt, sigmas)
    axs[i].plot(df_popG_countries.index, df_popG_countries[country], label="data")
    axs[i].plot(years, poly(years, *popt), label="fit")
    # plot error ranges with transparency
    axs[i].fill_between(years, lower, upper, alpha=0.5)
    axs[i].set_title(f"Polynomial Fit for {country}")
    axs[i].legend(loc="upper left")

# Adjust layout for neatness
plt.tight_layout()
plt.show()








