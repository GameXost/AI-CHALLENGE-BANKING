import matplotlib.pyplot as plt
import seaborn as sns
import umap
import numpy as np


class ClusterVisualizer:
	"""Визуализация кластеров с помощью UMAP"""

	def visualize_clusters(self, feature_matrix, info_df, feature_columns):
		"""
        Визуализация кластеров в 2D с помощью UMAP

        Параметры:
            feature_matrix: DataFrame с признаками
            info_df: DataFrame с информацией (включая ml_cluster)
            feature_columns: список названий признаков
        """
		try:
			print("\n📊 ВИЗУАЛИЗАЦИЯ КЛАСТЕРОВ...")

			X = feature_matrix[feature_columns].fillna(0)
			cluster_labels = info_df['ml_cluster'].values

			# UMAP для визуализации в 2D
			umap_vis = umap.UMAP(
				n_components=2,
				random_state=42,
				n_neighbors=15,
				min_dist=0.1
			)
			X_umap_2d = umap_vis.fit_transform(X)

			plt.figure(figsize=(14, 10))

			unique_clusters = np.unique(cluster_labels)
			colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))

			for i, cluster_id in enumerate(unique_clusters):
				if cluster_id == -1:
					color = 'gray'
					label = 'Шум'
					alpha = 0.3
					marker = 'x'
				else:
					color = colors[i]
					label = f'Кластер {cluster_id}'
					alpha = 0.7
					marker = 'o'

				mask = cluster_labels == cluster_id
				plt.scatter(
					X_umap_2d[mask, 0],
					X_umap_2d[mask, 1],
					c=[color],
					label=label,
					alpha=alpha,
					s=50,
					marker=marker
				)

			plt.title('Визуализация кластеров пользователей (UMAP)', fontsize=16)
			plt.xlabel('UMAP Component 1', fontsize=12)
			plt.ylabel('UMAP Component 2', fontsize=12)
			plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
			plt.grid(True, alpha=0.3)
			plt.tight_layout()

			# Сохранение
			try:
				plt.savefig('cluster_visualization.png', dpi=300, bbox_inches='tight')
				print("   ✅ Визуализация сохранена: cluster_visualization.png")
			except:
				pass

			plt.show()

		except Exception as e:
			print(f"❌ Ошибка визуализации: {e}")