import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#Reading Dataset
df = pd.read_csv('Data_Negara_HELP.csv')
df = df.rename(columns={'Kematian_anak':'Kematian Anak', 'Harapan_hidup':'Harapan Hidup','Jumlah_fertiliti':'Jumlah Fertilitas','GDPperkapita':'GDP per Kapita'})
display(df)
df.info()
df.describe()

#Univariate Analysis
plt.figure(figsize=(20,16))

for i in enumerate(df.describe().columns):
  plt.subplot(3,3, i[0]+1)
  sns.distplot(df[i[1]])
plt.show()

#Bivariate Analysis
anak_mati = df['Kematian Anak'] > 100
gdp_rendah = df['GDP per Kapita'] < 500

kematian_gdp = df[(anak_mati) & (gdp_rendah)].sort_values('Kematian Anak', ascending=False)
display(kematian_gdp)

#Heatmap
sns.heatmap(df.corr(), annot =True, fmt='.2g');

#Outliers
sns.boxplot('Kematian Anak', data=df)
sns.boxplot('GDP per Kapita', data=df)
sns.boxplot('Harapan Hidup', data=df)
sns.boxplot('Pendapatan', data=df)

def finding_outlier(df):
  Q1 = df.quantile(0.25)
  Q3 = df.quantile(0.75)
  IQR = Q3-Q1
  df_final = df[(df<(Q1-(1.5*IQR))) | (df>(Q3+(1.5*IQR)))]
  return df_final

print(finding_outlier(df['Kematian Anak']))
print(finding_outlier(df['Pendapatan']))
print(finding_outlier(df['Harapan Hidup']))
print(finding_outlier(df['Jumlah Fertilitas']))
print(finding_outlier(df['GDP per Kapita']))

def remove_outlier(df):
  Q1 = df.quantile(0.25)
  Q3 = df.quantile(0.75)
  IQR = Q3-Q1
  df_final = df[~((df<(Q1-(1.5*IQR))) | (df>(Q3+(1.5*IQR))))]
  return df_final

df2 = remove_outlier(df[['Kematian Anak','Pendapatan','Harapan Hidup','Jumlah Fertilitas','GDP per Kapita']])
df2.fillna(df2[['Kematian Anak','Pendapatan','Jumlah Fertilitas','GDP per Kapita']].max(),axis=0, inplace=True)
df2.fillna(df2[['Harapan Hidup']].min(),axis=0, inplace=True)

print(finding_outlier(df2['Kematian Anak']))
print(finding_outlier(df2['Pendapatan']))
print(finding_outlier(df2['Harapan Hidup']))
print(finding_outlier(df2['Jumlah Fertilitas']))
print(finding_outlier(df2['GDP per Kapita']))

#Clustering

#Elbow Method
wcss = []
for i in range(1,11):
  kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
  kmeans.fit(df2)
  wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title('Elbow method')
plt.xlabel('n_clusters')
plt.ylabel('wcss')

plt.show()

#Cluster 1 dan 2 merupakan clustering negara dari aspek kesehatan

#Cluster 1
data_cluster = df2[['Kematian Anak','Harapan Hidup']]
new_df = pd.DataFrame(data=data_cluster)
new_df['label1_kmeans'] = labels1

plt.figure(figsize=(12,8))

plt.scatter(new_df['Kematian Anak'][new_df.label1_kmeans == 0], new_df['Harapan Hidup'][new_df.label1_kmeans == 0], c='red', s=100, edgecolor='black')
plt.scatter(new_df['Kematian Anak'][new_df.label1_kmeans == 1], new_df['Harapan Hidup'][new_df.label1_kmeans == 1], c='yellow', s=100, edgecolor='black')

plt.scatter(kmeans1.cluster_centers_[:, 0],kmeans1.cluster_centers_[:, 1], c='k', s = 300)
plt.title('Country Cluster')
plt.xlabel('Kematian Anak')
plt.ylabel('Harapan Hidup')
plt.show()

#Cluster 2
kmeans2 = KMeans(n_clusters = 3, init='k-means++', random_state=42).fit(data_cluster)
labels2 = kmeans2.labels_

new_df['label2_kmeans'] = labels2
new_df

plt.figure(figsize=(12,8))

plt.scatter(new_df['Kematian Anak'][new_df.label2_kmeans == 0], new_df['Harapan Hidup'][new_df.label2_kmeans == 0], c='red', s=100, edgecolor='black')
plt.scatter(new_df['Kematian Anak'][new_df.label2_kmeans == 1], new_df['Harapan Hidup'][new_df.label2_kmeans == 1], c='yellow', s=100, edgecolor='black')
plt.scatter(new_df['Kematian Anak'][new_df.label2_kmeans == 2], new_df['Harapan Hidup'][new_df.label2_kmeans == 2], c='orange', s=100, edgecolor='black')

plt.scatter(kmeans2.cluster_centers_[:, 0],kmeans2.cluster_centers_[:, 1], c='k', s=300 )
plt.title('Country Cluster')
plt.xlabel('Kematian Anak')
plt.ylabel('Harapan Hidup')
plt.show()

#Silhoutter Score Cluster 1 & 2
from sklearn.metrics import silhouette_score
print(silhouette_score(data_cluster, labels=labels1))
print(silhouette_score(data_cluster, labels=labels2))

#List Negara Kesehatan
new_df['Negara'] = df['Negara']
cluster1 = new_df.label1_kmeans == 1
cluster2 = new_df.label2_kmeans == 1
new_df[(cluster1) & (cluster2)].sort_values('Kematian Anak', ascending=False).head()

#Cluster 3 dan 4 merupakan clustering negara dari aspek kesehatan

#Cluster 3
data_cluster3 = df2[['GDP per Kapita','Pendapatan']]

kmeans4 = KMeans(n_clusters = 2, random_state=42).fit(data_cluster3)
labels4 = kmeans4.labels_

new_df_eko = pd.DataFrame(data=data_cluster3)
new_df_eko['label4_kmeans'] = labels4

plt.figure(figsize=(12,8))

plt.scatter(new_df_eko['GDP per Kapita'][new_df_eko.label4_kmeans == 0], new_df_eko['Pendapatan'][new_df_eko.label4_kmeans == 0], c='red', s=100, edgecolor='black')
plt.scatter(new_df_eko['GDP per Kapita'][new_df_eko.label4_kmeans == 1], new_df_eko['Pendapatan'][new_df_eko.label4_kmeans == 1], c='yellow', s=100, edgecolor='black')

plt.scatter(kmeans4.cluster_centers_[:, 0],kmeans4.cluster_centers_[:, 1], c='k', s = 300)
plt.title('Country Cluster')
plt.xlabel('GDP per Kapita')
plt.ylabel('Pendapatan')
plt.show()

#Cluster 4
kmeans5 = KMeans(n_clusters = 3, init='k-means++', random_state=42).fit(data_cluster3)
labels5 = kmeans5.labels_

new_df_eko['label5_kmeans'] = labels5

plt.figure(figsize=(12,8))

plt.scatter(new_df_eko['GDP per Kapita'][new_df_eko.label5_kmeans == 0], new_df_eko['Pendapatan'][new_df_eko.label5_kmeans == 0], c='red', s=100, edgecolor='black')
plt.scatter(new_df_eko['GDP per Kapita'][new_df_eko.label5_kmeans == 1], new_df_eko['Pendapatan'][new_df_eko.label5_kmeans == 1], c='yellow', s=100, edgecolor='black')
plt.scatter(new_df_eko['GDP per Kapita'][new_df_eko.label5_kmeans == 2], new_df_eko['Pendapatan'][new_df_eko.label5_kmeans == 2], c='orange', s=100, edgecolor='black')

plt.scatter(kmeans5.cluster_centers_[:, 0],kmeans5.cluster_centers_[:, 1], c='k', s=300 )
plt.title('Country Cluster')
plt.xlabel('GDP per Kapita')
plt.ylabel('Pendapatan')
plt.show()

#Silhoutte Score Cluster 3 & 4
from sklearn.metrics import silhouette_score
print(silhouette_score(data_cluster3, labels=labels4))
print(silhouette_score(data_cluster3, labels=labels5))

#List Negara Ekonomi
new_df_eko['Negara'] = df['Negara']
cluster4 = new_df_eko.label4_kmeans == 0
cluster5 = new_df_eko.label5_kmeans == 0
new_df_eko[(cluster4) & (cluster5)].sort_values('GDP per Kapita', ascending=True).head()

#KESIMPULAN (gabungan list negara dari bagian kesehatan dan bagian ekonomi)
kes = new_df[(cluster1) & (cluster2)].sort_values('Kematian Anak', ascending=False).head(10)
eko = new_df_eko[(cluster4) & (cluster5)].sort_values('GDP per Kapita', ascending=True).head(10)

overall = pd.merge(kes, eko, on='Negara', how='inner')
overall = overall[['Negara','Kematian Anak','Harapan Hidup','GDP per Kapita','Pendapatan']]
display(overall)
