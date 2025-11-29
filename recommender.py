import pandas as pd
import numpy as np


class SmartProductRecommender:
	"""
    Система рекомендаций на основе интерпретированных кластеров

    ЛОГИКА:
    1. Берем профиль кластера из cluster_analyzer
    2. На основе ключевых метрик подбираем продукты
    3. Генерируем персонализированные рекомендации
    """

	def __init__(self):
		# Расширенный каталог продуктов
		self.product_catalog = {
			'credit_cards': "Кредитная карта",
			'debit_cashback': "Дебетовая карта с кешбэком",
			'premium_card': "Премиальная карта",
			'youth_card': "Молодежная карта",
			'express_credit': "Экспресс-кредит",
			'mortgage': "Ипотека",
			'auto_loan': "Автокредит",
			'deposit': "Вклад с высокой ставкой",
			'savings_account': "Накопительный счет",
			'investment_portfolio': "Инвестиционный портфель",
			'pension_savings': "Пенсионный накопительный план",
			'insurance': "Страхование",
			'life_insurance': "Страхование жизни",
			'premium_service': "Премиальное обслуживание",
			'mobile_bank': "Мобильный банк",
			'business_account': "Бизнес-счет"
		}

		# Стратегии для каждого типа кластера
		self.cluster_strategies = {
			'Премиум-клиенты': [
				'premium_service', 'investment_portfolio', 'premium_card',
				'life_insurance', 'pension_savings'
			],
			'Молодые активные': [
				'youth_card', 'debit_cashback', 'mobile_bank',
				'savings_account', 'express_credit'
			],
			'Исследователи': [
				'deposit', 'investment_portfolio', 'savings_account',
				'insurance', 'debit_cashback'
			],
			'Импульсивные покупатели': [
				'express_credit', 'credit_cards', 'debit_cashback',
				'insurance', 'mobile_bank'
			],
			'Кредитно-ориентированные': [
				'mortgage', 'auto_loan', 'credit_cards',
				'insurance', 'express_credit'
			],
			'Накопители': [
				'deposit', 'savings_account', 'pension_savings',
				'investment_portfolio', 'life_insurance'
			],
			'Неактивные': [
				'mobile_bank', 'debit_cashback', 'savings_account',
				'insurance', 'youth_card'
			],
			'Консервативные': [
				'deposit', 'life_insurance', 'pension_savings',
				'savings_account', 'premium_service'
			],
			'Средний сегмент': [
				'debit_cashback', 'savings_account', 'insurance',
				'credit_cards', 'mobile_bank'
			]
		}

	def generate_recommendations(self, feature_matrix, info_df, cluster_profiles):
		"""
        Генерация рекомендаций на основе кластеров

        Параметры:
            feature_matrix: DataFrame с признаками
            info_df: DataFrame с информацией о пользователях (включая ml_cluster)
            cluster_profiles: словарь профилей кластеров из cluster_analyzer
        """
		print(f"\n🎁 ГЕНЕРАЦИЯ ПЕРСОНАЛИЗИРОВАННЫХ РЕКОМЕНДАЦИЙ")
		print("-" * 70)

		cluster_labels = info_df['ml_cluster'].values
		recommendations = []

		for i, (user_id, cluster_id) in enumerate(zip(info_df['user_id'], cluster_labels)):
			# Получаем профиль кластера
			cluster_profile = cluster_profiles.get(cluster_id)

			if cluster_id == -1:
				# Пользователь в шуме - даем базовые рекомендации
				cluster_type = "Неопределенный"
				recommended_products = self._get_default_products()
				confidence = 0.3
			elif cluster_profile is None:
				cluster_type = f"Кластер_{cluster_id}"
				recommended_products = self._get_default_products()
				confidence = 0.5
			else:
				cluster_type = cluster_profile['type']
				recommended_products = self._get_products_for_cluster(cluster_profile)
				confidence = self._calculate_confidence(cluster_profile, feature_matrix.iloc[i])

			# Преобразуем коды продуктов в названия
			product_names = [self.product_catalog.get(p, p) for p in recommended_products]

			recommendations.append({
				'user_id': int(user_id),
				'cluster_id': int(cluster_id) if cluster_id != -1 else -1,
				'user_category': cluster_type,
				'category_confidence': confidence,
				'top_recommendation': product_names[0] if product_names else "Дебетовая карта",
				'recommended_products': product_names[:5]  # Топ-5
			})

		print(f"✅ Создано {len(recommendations)} персонализированных рекомендаций")

		return recommendations

	def _get_products_for_cluster(self, cluster_profile):
		"""Подбор продуктов на основе профиля кластера"""
		cluster_type = cluster_profile['type']

		# Базовая стратегия
		base_products = self.cluster_strategies.get(cluster_type, self._get_default_products())

		# Дополнительная персонализация на основе метрик
		key_metrics = cluster_profile.get('key_metrics', {})

		# Добавляем кредитные продукты, если высокая склонность
		if key_metrics.get('credit_products_affinity', 0) > 0.4:
			if 'credit_cards' not in base_products:
				base_products.insert(1, 'credit_cards')

		# Добавляем сберегательные продукты, если высокая склонность
		if key_metrics.get('saving_products_affinity', 0) > 0.4:
			if 'savings_account' not in base_products:
				base_products.insert(1, 'savings_account')

		# Добавляем мобильный банк для цифровых пользователей
		if key_metrics.get('engagement_level', 0) > 0.6:
			if 'mobile_bank' not in base_products:
				base_products.append('mobile_bank')

		return base_products[:5]  # Возвращаем топ-5

	def _get_default_products(self):
		"""Базовый набор продуктов"""
		return ['debit_cashback', 'savings_account', 'insurance', 'mobile_bank', 'credit_cards']

	def _calculate_confidence(self, cluster_profile, user_features):
		"""
        Расчет уверенности в рекомендации

        Основан на:
        - Размере кластера (больше = стабильнее)
        - Четкости профиля (насколько сильно отличаются ключевые признаки)
        """
		cluster_size = cluster_profile.get('size', 0)

		# Базовая уверенность от размера кластера
		size_confidence = min(1.0, cluster_size / 50)  # Max при 50+ пользователях

		# Уверенность от четкости признаков
		top_features = cluster_profile.get('top_features', [])
		if top_features:
			avg_importance = np.mean([importance for _, importance in top_features])
			feature_confidence = min(1.0, avg_importance / 3)  # Max при важности ~3
		else:
			feature_confidence = 0.5

		# Итоговая уверенность
		confidence = (size_confidence * 0.4 + feature_confidence * 0.6)

		return round(confidence, 2)

	def show_results(self, recommendations):
		"""Визуализация результатов рекомендательной системы"""
		print(f"\n📊 АНАЛИЗ РЕКОМЕНДАЦИЙ")
		print("=" * 70)

		if not recommendations:
			print("❌ Нет рекомендаций для показа")
			return

		# 1. ОБЩАЯ СТАТИСТИКА
		total_users = len(recommendations)
		unique_products = set()
		for rec in recommendations:
			unique_products.update(rec['recommended_products'])

		print(f"\n🎯 ОБЩИЕ ПОКАЗАТЕЛИ:")
		print(f"   Пользователей обработано: {total_users}")
		print(f"   Уникальных продуктов: {len(unique_products)}")

		# Средняя уверенность
		avg_confidence = np.mean([rec['category_confidence'] for rec in recommendations])
		print(f"   Средняя уверенность: {avg_confidence:.2f}")

		# 2. РАСПРЕДЕЛЕНИЕ ПО КЛАСТЕРАМ
		cluster_groups = {}
		for rec in recommendations:
			cluster = rec['user_category']
			if cluster not in cluster_groups:
				cluster_groups[cluster] = []
			cluster_groups[cluster].append(rec)

		print(f"\n👥 РАСПРЕДЕЛЕНИЕ ПО СЕГМЕНТАМ:")
		for cluster, recs in sorted(cluster_groups.items(), key=lambda x: len(x[1]), reverse=True):
			size = len(recs)
			percentage = (size / total_users) * 100
			avg_conf = np.mean([r['category_confidence'] for r in recs])
			print(f"   {cluster}: {size} ({percentage:.1f}%) - уверенность {avg_conf:.2f}")

		# 3. ТОП ПРОДУКТОВ
		all_recommendations = []
		for rec in recommendations:
			all_recommendations.extend(rec['recommended_products'])

		product_counts = {}
		for product in all_recommendations:
			product_counts[product] = product_counts.get(product, 0) + 1

		print(f"\n🏆 ТОП-10 РЕКОМЕНДУЕМЫХ ПРОДУКТОВ:")
		sorted_products = sorted(product_counts.items(), key=lambda x: x[1], reverse=True)
		for i, (product, count) in enumerate(sorted_products[:10], 1):
			percentage = (count / len(all_recommendations)) * 100
			print(f"   {i}. {product}: {count} раз ({percentage:.1f}%)")

		# 4. ПРИМЕРЫ РЕКОМЕНДАЦИЙ
		print(f"\n📋 ПРИМЕРЫ РЕКОМЕНДАЦИЙ ПО СЕГМЕНТАМ:")
		shown_clusters = set()
		for rec in recommendations:
			cluster = rec['user_category']
			if cluster not in shown_clusters and len(shown_clusters) < 5:
				print(f"\n   🎯 {cluster} (user_id: {rec['user_id']})")
				print(f"      Уверенность: {rec['category_confidence']:.2f}")
				print(f"      Топ продукт: {rec['top_recommendation']}")
				print(f"      Все рекомендации: {', '.join(rec['recommended_products'][:3])}")
				shown_clusters.add(cluster)

		# 5. КАТЕГОРИИ ПРОДУКТОВ
		product_categories = {
			'Карты': ['Кредитная карта', 'Дебетовая карта с кешбэком', 'Премиальная карта', 'Молодежная карта'],
			'Кредиты': ['Экспресс-кредит', 'Ипотека', 'Автокредит'],
			'Накопления': ['Вклад с высокой ставкой', 'Накопительный счет', 'Пенсионный накопительный план'],
			'Инвестиции': ['Инвестиционный портфель'],
			'Страхование': ['Страхование', 'Страхование жизни'],
			'Сервисы': ['Премиальное обслуживание', 'Мобильный банк', 'Бизнес-счет']
		}

		category_counts = {}
		for category, products in product_categories.items():
			count = sum(product_counts.get(p, 0) for p in products)
			category_counts[category] = count

		print(f"\n📈 РАСПРЕДЕЛЕНИЕ ПО КАТЕГОРИЯМ:")
		total_recs = len(all_recommendations)
		for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
			percentage = (count / total_recs) * 100
			print(f"   {category}: {count} ({percentage:.1f}%)")

		print(f"\n{'=' * 70}")
		print(f"✅ Анализ завершен успешно!")