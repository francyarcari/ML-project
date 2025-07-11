import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#Caricamento del dataset dei customer BankChurners.csv
df = pd.read_csv('BankChurners.csv')

#Visualizzazione struttura del dataset

df.head()
df.info()


# Removing coloumns 0, 1. 21 e 22
df = df.drop(columns=['CLIENTNUM','Attrition_Flag', 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1', 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'])   

#Checking the distribution of  categorical variables

print(df['Gender'].value_counts(), end="\n\n")
print(df['Marital_Status'].value_counts(), end="\n\n")
print(df['Education_Level'].value_counts(), end="\n\n")
print(df['Income_Category'].value_counts(),end="\n\n") 
print(df['Card_Category'].value_counts())

#Ora esploriamo le variabili numeriche per capire le loro distribuzioni (media, deviazione standard, quantili)
#Per avere una tabella più compatta dividiamo le variabili in due gruppi

cols1 = ['Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count','Months_Inactive_12_mon',
         'Contacts_Count_12_mon', 'Credit_Limit']

cols2 = ['Total_Revolving_Bal','Avg_Open_To_Buy','Total_Amt_Chng_Q4_Q1', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1',
         'Avg_Utilization_Ratio']

# Statistiche descrittive per il primo gruppo
print("Statistiche - Gruppo 1:\n")
print(df[cols1].describe().T.round(2))

# Statistiche descrittive per il secondo gruppo
print("\nStatistiche - Gruppo 2:\n")
print(df[cols2].describe().T.round(2))


#To make it easier to spot patterns, let's visualize the distribution of each variable using histograms:

# Seleziona colonne numeriche
df_numeric = df.select_dtypes(include='number')

# Configura layout
num_cols = df_numeric.shape[1]
cols_per_row = 4  # Più colonne per rendere i singoli grafici più piccoli
rows = (num_cols + cols_per_row - 1) // cols_per_row

# A4 verticale = 8.27 x 11.69 pollici
fig, axes = plt.subplots(rows, cols_per_row, figsize=(8.27, 11.69))
axes = axes.flatten()

# Traccia istogrammi
for i, col in enumerate(df_numeric.columns):
    sns.histplot(df_numeric[col], bins=20, kde=False, ax=axes[i], color='steelblue')
    axes[i].set_title(col, fontsize=8)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')
    axes[i].tick_params(axis='x', labelrotation=45, labelsize=6)
    axes[i].tick_params(axis='y', labelsize=6)

# Rimuove assi vuoti
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Imposta layout compatto
plt.tight_layout()
plt.suptitle("Istogrammi Variabili Numeriche", fontsize=12)
plt.subplots_adjust(top=0.93)
plt.show()

#Mappa di correlazione
#Codice creare la mappa di correlazione
numerical_df = df.select_dtypes(include=[np.number])

# Calcola la matrice di correlazione
corr_matrix = numerical_df.corr()

#Crea una maschera per nascondere il triangolo superiore
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Imposta le dimensioni della figura e visualizza l'heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    center=0,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)
plt.title("Mappa di correlazione tra le variabili numeriche", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

#Trasformazione delle variabili di categoria in variabili numeriche per applicare l'algoritmo di clustering K-means
#Crea una copia del DataFrame per non modificare l'originale

df_encoded = df.copy()

# Istanza del codificatore
label_encoder = LabelEncoder()

# Codifica automatica delle colonne categoriche
for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object' or df_encoded[col].dtype.name == 'category':
        df_encoded[col] = label_encoder.fit_transform(df_encoded[col])

# Verifica: controlla i primi valori e i tipi
print(df_encoded.dtypes)
print(df_encoded.head())

#Scaling del dataset

X = df_encoded.copy()

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)


#Utilizziamo l'elbow method per trovare il numero ottimo di K cluster

X = pd.DataFrame(X_scaled_df)
inertias = []
for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42)  # Fissiamo il seed per stabilità
    y = model.fit_predict(X)
    inertias.append(model.inertia_)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertias, marker='o', linestyle='--', color='darkblue')
plt.title('Elbow Method - Inerzia vs Numero di Cluster')
plt.xlabel('Numero di Cluster (k)')
plt.ylabel('Inerzia')
plt.xticks(range(1, 11))
plt.grid(True)
plt.tight_layout()
plt.show()


#Utilizziamo l'elbow method senza K-Means++ per trovare il numero ottimo di K cluster (per confronto con quello con K-Means++)
inertias = []
# Prova da 1 a 10 cluster usando init='random'
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='random', n_init=10, random_state=42)
    kmeans.fit(X_scaled_df)
    inertias.append(kmeans.inertia_)

# Grafico Elbow
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertias, marker='o')
plt.title("Elbow Method (init='random')")
plt.xlabel("Numero di cluster (k)")
plt.ylabel("Inertia")
plt.grid(True)
plt.tight_layout()
plt.show()

# Calcolo del Silhouette Score vs Numero di Cluster (k)
silhouette_scores = []
ks = range(2, 11)

for k in ks:
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X)
    silhouette_scores.append(silhouette_score(X, labels))

plt.figure(figsize=(10, 6))
plt.plot(ks, silhouette_scores, marker='o', linestyle='--', color='green')
plt.title("Silhouette Score vs Numero di Cluster")
plt.xlabel("Numero di Cluster (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()


# PCA a 3 componenti

pca = PCA(n_components=3)
X_reduced = pca.fit_transform(X_scaled_df)

X = pd.DataFrame(X_reduced)
inertias = []
for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42)  # Fissiamo il seed per stabilità
    y = model.fit_predict(X)
    inertias.append(model.inertia_)


#Elbow method con PCA
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertias, marker='o', linestyle='--', color='darkblue')
plt.title('Elbow Method dopo PCA')
plt.xlabel('Numero di Cluster (k)')
plt.ylabel('Inerzia')
plt.xticks(range(1, 11))
plt.grid(True)
plt.tight_layout()
plt.show()

# Calcolo del Silhouette Score dopo PCA
silhouette_scores = []
ks = range(2, 11)

for k in ks:
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X)
    silhouette_scores.append(silhouette_score(X, labels))

plt.figure(figsize=(10, 6))
plt.plot(ks, silhouette_scores, marker='o', linestyle='--', color='green')
plt.title("Silhouette Score dopo PCA")
plt.xlabel("Numero di Cluster (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()

#Ora che abbiamo definito il numero di cluster ottimali, passiamo a costruire il nostro modello k-means

k = int(input("Inserisci il numero di cluster ottimale: (2)"))
print(f"Costruzione del modello K-Means con k = {k} cluster...")
    
# K-Means finale su dati ridotti con PCA 

kmeans_final = KMeans(n_clusters=k, random_state=42)

labels = kmeans_final.fit_predict(X_reduced)
labels = labels + 1


# === 1. DataFrame con dati PCA + cluster (per visualizzazioni)
df_pca_clusters = pd.DataFrame(X_reduced, columns=['PC1', 'PC2', 'PC3'])
df_pca_clusters['Cluster'] = ['Cluster ' + str(label) for label in labels]


print("Clustering completato. Ecco un'anteprima:")
print(df_pca_clusters.head())


# Calcolo delle medie per ciascuna variabile, raggruppate per cluster

X_scaled_df['Cluster'] = labels

cluster_means = X_scaled_df.groupby('Cluster').mean().T  # Trasposto: variabili su righe

# Plot verticale: barre affiancate per Cluster 1 e 2
plt.figure(figsize=(12, 18))
cluster_means.plot(kind='bar', figsize=(10, 18), width=0.7)

plt.title('Media (standardizzata) per variabile e per cluster')
plt.ylabel('Media standardizzata')
plt.xlabel('Variabili')
plt.xticks(rotation=90)
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()


#Grafico di dispersione

X_reduced_2D = X_reduced[:, :2]

kmeans_final = KMeans(n_clusters=k, random_state=42)
labels = kmeans_final.fit_predict(X_reduced_2D)
centroids = kmeans_final.cluster_centers_

# Plot
plt.figure(figsize=(10, 7))
colors = ['blue', 'orange', 'green', 'red', 'purple']  # Per più cluster

# Disegna ciascun cluster
for i in range(k):
    cluster_points = X_reduced_2D[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                s=40, c=colors[i], label=f'Cluster {i+1}', alpha=0.6)

# Centroidi in nero
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='o', s=200, c='black', label='Centroidi')

# Titoli e assi leggibili
plt.title('Visualizzazione dei Cluster su PCA (2D)', fontsize=16)
plt.xlabel('PC1', fontsize=12)
plt.ylabel('PC2', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

