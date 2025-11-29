import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import umap

try:
	import hdbscan

	HDBSCAN_AVAILABLE = True
except ImportError:
	HDBSCAN_AVAILABLE = False
	print("⚠️ HDBSCAN не установлен. Установите: pip install hdbscan")
	print("   Будет использован DBSCAN как fallback")


class AutoClusterAnalyzer:
	"""
    Автоматический анализ кластеров с использованием HDBSCAN

    ПРЕИМУЩЕСТВА ПЕРЕД DBSCAN:
    - Автоматически определяет количество кластеров
    - Не требует ручного подбора параметров eps/min_samples
    - Находит кластеры произвольной формы
    - Помечает выбросы как шум

    АРХИТЕКТУРА:
    1. StandardScaler - нормализация признаков
    2. UMAP - снижение размерности 104 → 10-15 компонент
    3. HDBSCAN - автоматическая кластеризация
    4. Интерпретация - создание профилей кластеров
    """

	def __init__(self):
		self.scaler = StandardScaler()
		self.umap_reducer = None
		self.clusterer = None
		self.cluster_profiles = {}

	def train_clustering_model(self, feature_matrix, info_df, feature_columns):
		"""
        Обучение модели кластеризации

        Параметры:
            feature_matrix: DataFrame с признаками
            info_df: DataFrame с информацией о пользователях
            feature_columns: список названий признаков
        """
		print(f"\n🔬 АВТОМАТИЧЕСКАЯ КЛАСТЕРИЗАЦИЯ (HDBSCAN + UMAP)")
		print("-" * 70)

		if len(feature_matrix) < 30:
			print("⚠️ Недостаточно данных для кластеризации (<30 пользователей)")
			info_df['ml_cluster'] = 0
			return info_df

		# 1. ПОДГОТОВКА ДАННЫХ
		X = feature_matrix[feature_columns].fillna(0)
		X_scaled = self.scaler.fit_transform(X)

		print(f"✅ Подготовка данных:")
		print(f"   Пользователей: {len(X)}")
		print(f"   Признаков: {len(feature_columns)}")

		# 2. СНИЖЕНИЕ РАЗМЕРНОСТИ С UMAP
		print(f"\n🧬 Снижение размерности (UMAP)...")
		n_components = min(15, X_scaled.shape[1] - 1)

		self.umap_reducer = umap.UMAP(
			n_components=n_components,
			n_neighbors=15,
			min_dist=0.0,  # Для кластеризации лучше 0.0
			metric='euclidean',
			random_state=42
		)
		X_reduced = self.umap_reducer.fit_transform(X_scaled)
		print(f"   Размерность: {X_scaled.shape[1]} → {X_reduced.shape[1]}")

		# 3. КЛАСТЕРИЗАЦИЯ С HDBSCAN
		if not HDBSCAN_AVAILABLE:
			print("\n❌ HDBSCAN недоступен! Используется DBSCAN...")
			from sklearn.cluster import DBSCAN
			dbscan = DBSCAN(eps=0.5, min_samples=10)
			cluster_labels = dbscan.fit_predict(X_reduced)
		else:
			print(f"\n🎯 Кластеризация (HDBSCAN)...")

			# Автоматический расчет min_cluster_size
			# Правило: 2-5% от общего количества или минимум 10
			min_cluster_size = max(10, int(len(X_reduced) * 0.03))
			min_samples = max(5, int(min_cluster_size * 0.5))

			self.clusterer = hdbscan.HDBSCAN(
				min_cluster_size=min_cluster_size,
				min_samples=min_samples,
				metric='euclidean',
				cluster_selection_method='eom',  # Excess of Mass
				prediction_data=True
			)

			cluster_labels = self.clusterer.fit_predict(X_reduced)

			print(f"   Параметры:")
			print(f"      min_cluster_size: {min_cluster_size}")
			print(f"      min_samples: {min_samples}")

		# 4. АНАЛИЗ РЕЗУЛЬТАТОВ
		unique_labels = np.unique(cluster_labels)
		n_clusters = len(unique_labels[unique_labels != -1])
		noise_points = np.sum(cluster_labels == -1)
		noise_ratio = noise_points / len(cluster_labels)

		print(f"\n📊 РЕЗУЛЬТАТЫ КЛАСТЕРИЗАЦИИ:")
		print(f"   Найдено кластеров: {n_clusters}")
		print(f"   Шум (выбросы): {noise_points} ({noise_ratio:.1%})")

		# 5. МЕТРИКИ КАЧЕСТВА
		if n_clusters > 1:
			mask = cluster_labels != -1
			if np.sum(mask) > n_clusters:
				silhouette = silhouette_score(X_reduced[mask], cluster_labels[mask])
				davies_bouldin = davies_bouldin_score(X_reduced[mask], cluster_labels[mask])
				calinski = calinski_harabasz_score(X_reduced[mask], cluster_labels[mask])

				print(f"\n🎯 МЕТРИКИ КАЧЕСТВА:")
				print(f"   Silhouette Score: {silhouette:.3f} (выше = лучше, 0.3-0.7 хорошо)")
				print(f"   Davies-Bouldin Index: {davies_bouldin:.3f} (ниже = лучше)")
				print(f"   Calinski-Harabasz Score: {calinski:.1f} (выше = лучше)")

		# 6. РАСПРЕДЕЛЕНИЕ ПО КЛАСТЕРАМ
		print(f"\n📈 РАСПРЕДЕЛЕНИЕ ПОЛЬЗОВАТЕЛЕЙ:")
		for cluster_id in sorted(unique_labels):
			if cluster_id == -1:
				continue
			cluster_size = np.sum(cluster_labels == cluster_id)
			percentage = (cluster_size / len(cluster_labels)) * 100
			print(f"   Кластер {cluster_id}: {cluster_size} пользователей ({percentage:.1f}%)")

		if noise_points > 0:
			print(f"   Шум: {noise_points} пользователей ({noise_ratio:.1%})")

		# 7. ИНТЕРПРЕТАЦИЯ КЛАСТЕРОВ
		self.cluster_profiles = self._interpret_clusters(
			X_scaled, feature_matrix, cluster_labels, feature_columns
		)

		# 8. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
		info_df['ml_cluster'] = cluster_labels

		return info_df

	def _interpret_clusters(self, X_scaled, feature_matrix, cluster_labels, feature_columns):
		"""
        Интерпретация кластеров через анализ средних значений признаков

        Создает профиль каждого кластера с:
        - Топ-5 отличительных признаков
        - Средние значения ключевых метрик
        - Человекочитаемое название
        """
		print(f"\n🔍 ИНТЕРПРЕТАЦИЯ КЛАСТЕРОВ:")
		print("-" * 70)

		cluster_profiles = {}
		unique_clusters = np.unique(cluster_labels)

		# Ключевые признаки для интерпретации
		key_features = [
			'age_numeric', 'socdem_cluster', 'engagement_level',
			'financial_activity', 'impulse_score', 'search_ratio',
			'mean_price', 'payment_count', 'total_actions',
			'weekend_ratio', 'product_diversity', 'credit_products_affinity',
			'saving_products_affinity', 'spending_trend'
		]

		# Фильтруем только существующие признаки
		available_key_features = [f for f in key_features if f in feature_columns]

		for cluster_id in unique_clusters:
			if cluster_id == -1:  # Пропускаем шум
				continue

			cluster_mask = cluster_labels == cluster_id
			cluster_size = np.sum(cluster_mask)

			if cluster_size < 3:  # Слишком маленький кластер
				continue

			# Получаем данные кластера
			cluster_data = feature_matrix[feature_columns].iloc[cluster_mask]

			# Средние значения по ключевым признакам
			cluster_means = {}
			for feature in available_key_features:
				cluster_means[feature] = cluster_data[feature].mean()

			# Находим топ-5 отличительных признаков
			# (признаки, где среднее кластера сильно отличается от общего среднего)
			all_means = feature_matrix[feature_columns].mean()
			feature_importance = {}

			for feature in feature_columns:
				if feature_matrix[feature].std() > 0:  # Избегаем деления на 0
					cluster_mean = cluster_data[feature].mean()
					overall_mean = all_means[feature]
					overall_std = feature_matrix[feature].std()

					# Z-score различия
					z_score = abs(cluster_mean - overall_mean) / overall_std
					feature_importance[feature] = z_score

			# Топ-5 признаков
			top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]

			# Классификация типа кластера
			cluster_type = self._classify_cluster_type(cluster_means)

			# Сохраняем профиль
			cluster_profiles[cluster_id] = {
				'size': cluster_size,
				'type': cluster_type,
				'key_metrics': cluster_means,
				'top_features': top_features,
			}

			# Вывод информации
			print(f"\n📋 Кластер {cluster_id} ({cluster_size} пользователей) - '{cluster_type}'")
			print(f"   Топ-5 отличительных признаков:")
			for feature, importance in top_features:
				value = cluster_data[feature].mean()
				print(f"      • {feature}: {value:.3f} (важность: {importance:.2f})")

		return cluster_profiles

	def _classify_cluster_type(self, cluster_means):
		"""
        Определение типа кластера на основе средних значений

        Возвращает человекочитаемое название:
        - "Молодые активные"
        - "Премиум-клиенты"
        - "Исследователи"
        - "Консервативные"
        - и т.д.
        """
		# Безопасное извлечение значений
		age = cluster_means.get('age_numeric', 3)
		financial = cluster_means.get('financial_activity', 0)
		engagement = cluster_means.get('engagement_level', 0)
		impulse = cluster_means.get('impulse_score', 0)
		search = cluster_means.get('search_ratio', 0)
		mean_price = cluster_means.get('mean_price', 0)
		credit_affinity = cluster_means.get('credit_products_affinity', 0)
		saving_affinity = cluster_means.get('saving_products_affinity', 0)

		# Логика классификации
		if mean_price > 6000 and financial > 0.5:
			return "Премиум-клиенты"
		elif age < 2 and engagement > 0.4:
			return "Молодые активные"
		elif search > 0.5 and impulse < 0.3:
			return "Исследователи"
		elif impulse > 0.6:
			return "Импульсивные покупатели"
		elif credit_affinity > 0.3:
			return "Кредитно-ориентированные"
		elif saving_affinity > 0.3:
			return "Накопители"
		elif engagement < 0.2 and financial < 0.2:
			return "Неактивные"
		elif age > 3.5:
			return "Консервативные"
		else:
			return "Средний сегмент"

	def get_cluster_profile(self, cluster_id):
		"""Получить профиль конкретного кластера"""
		return self.cluster_profiles.get(cluster_id, None)

	def predict_cluster(self, new_user_features):
		"""
        Предсказание кластера для нового пользователя

        Требуется обученная модель HDBSCAN с prediction_data=True
        """
		if self.clusterer is None or not HDBSCAN_AVAILABLE:
			raise ValueError("Модель не обучена или HDBSCAN недоступен")

		# Нормализация и снижение размерности
		X_scaled = self.scaler.transform(new_user_features.reshape(1, -1))
		X_reduced = self.umap_reducer.transform(X_scaled)

		# Предсказание
		cluster_label, strength = hdbscan.approximate_predict(self.clusterer, X_reduced)

		return cluster_label[0], strength[0]