import pandas as pd
import numpy as np
from datetime import timedelta
from scipy import stats


class AdvancedFeatureEngineer:
	"""
	Расширенный инженеринг признаков: ~100 фичей вместо 36

	НОВЫЕ БЛОКИ:
	- Временные паттерны (день недели, час, сезонность)
	- Частотные характеристики по категориям
	- Топ/антитоп категории продуктов
	- Статистики временных рядов
	- Продвинутые финансовые метрики
	"""

	def __init__(self):
		self.feature_columns = []
		self.product_categories = ['credit_cards', 'deposits', 'investments',
								   'insurance', 'mortgage', 'auto_loans']

	def create_feature_vectors(self, users_df, market_df, payments_df, retail_df, sample_size=200):
		print("\n🧬 СОЗДАНИЕ РАСШИРЕННОГО НАБОРА ПРИЗНАКОВ (~100 фичей)")
		print("-" * 70)

		if len(market_df) == 0:
			print("❌ Нет данных маркетплейса")
			return pd.DataFrame(), pd.DataFrame()

		# Фильтрация по времени (последние 90 дней)
		if 'timestamp' in market_df.columns:
			recent_date = market_df['timestamp'].max()
			cutoff_date = recent_date - timedelta(days=90)
			market_df = market_df[market_df['timestamp'] >= cutoff_date]
			print(f"✅ Данные за последние 90 дней: {len(market_df)} событий")

		active_users = market_df['user_id'].unique()
		if len(active_users) == 0:
			print("❌ Нет активных пользователей")
			return pd.DataFrame(), pd.DataFrame()

		sample_users = np.random.choice(active_users, min(sample_size, len(active_users)), replace=False)
		print(f"👥 Обработка {len(sample_users)} пользователей...")

		vectors, user_info = self._process_users(sample_users, users_df, market_df, payments_df)

		if not vectors:
			print("❌ Не удалось создать векторы")
			return pd.DataFrame(), pd.DataFrame()

		feature_matrix = pd.DataFrame(vectors)
		info_df = pd.DataFrame(user_info)
		self.feature_columns = [col for col in feature_matrix.columns if col != 'user_id']

		print(f"\n✅ Создано векторов: {len(vectors)}")
		print(f"📊 Признаков на пользователя: {len(self.feature_columns)}")
		print(f"🎯 Целевое количество: 100+")

		return feature_matrix, info_df

	def _process_users(self, sample_users, users_df, market_df, payments_df):
		vectors = []
		user_info = []

		for i, user_id in enumerate(sample_users):
			try:
				vector, info = self._compute_features(user_id, users_df, market_df, payments_df)
				if vector is not None:
					vectors.append(vector)
					user_info.append(info)

				if (i + 1) % 50 == 0:
					print(f"   ⏳ Обработано: {i + 1}/{len(sample_users)}")

			except Exception as e:
				continue

		return vectors, user_info

	def _compute_features(self, user_id, users_df, market_df, payments_df):
		"""
		ПОЛНЫЙ НАБОР ПРИЗНАКОВ:
		1. Демографические (7)
		2. Финансовые базовые (12)
		3. Финансовые продвинутые (15) ← НОВОЕ
		4. Поведенческие (12)
		5. Временные паттерны (25) ← НОВОЕ
		6. Продуктовые предпочтения (18) ← РАСШИРЕНО
		7. Частотные характеристики (10) ← НОВОЕ
		8. Дополнительные (5)

		ИТОГО: ~104 признака
		"""
		try:
			# Получаем данные пользователя
			user_data = users_df[users_df['user_id'] == user_id]
			if len(user_data) == 0:
				return None, None
			user_data = user_data.iloc[0]

			user_market = market_df[market_df['user_id'] == user_id].copy()
			user_payments = payments_df[payments_df['user_id'] == user_id].copy() if len(
				payments_df) > 0 else pd.DataFrame()

			if len(user_market) == 0:
				return None, None

			# Базовые агрегации
			total_actions = len(user_market)
			action_types = user_market['action_type'].value_counts()
			categories = user_market['subdomain'].value_counts()
			product_categories = user_market[
				'product_category'].value_counts() if 'product_category' in user_market.columns else pd.Series()

			# Вспомогательные функции
			def safe_ratio(num, denom):
				return num / denom if denom > 0 else 0.0

			def safe_get(series, key, default=0):
				return series.get(key, default) if len(series) > 0 else default

			# ============================================================
			# 1. ДЕМОГРАФИЧЕСКИЕ ПРИЗНАКИ (7)
			# ============================================================
			age_group = user_data.get('age_group', 'unknown')
			if age_group == 'unknown':
				socdem = user_data.get('socdem_cluster', 0)
				if socdem < 3:
					age_group = '18-25'
				elif socdem < 6:
					age_group = '26-35'
				elif socdem < 8:
					age_group = '36-45'
				else:
					age_group = '46-55'

			demographic_features = {
				'user_id': user_id,
				'socdem_cluster': user_data['socdem_cluster'],
				'region': user_data.get('region', 0),
				'is_young': 1 if age_group in ['18-25', '26-35'] else 0,
				'is_family': 1 if age_group in ['36-45'] else 0,
				'is_mature': 1 if age_group in ['46-55', '55+'] else 0,
				'age_numeric': {'18-25': 1, '26-35': 2, '36-45': 3, '46-55': 4, '55+': 5}.get(age_group, 3),
			}

			# ============================================================
			# 2. ФИНАНСОВЫЕ ПРИЗНАКИ - БАЗОВЫЕ (12)
			# ============================================================
			payment_features = {}
			if len(user_payments) > 0:
				payment_prices = user_payments['price']
				payment_count = len(user_payments)
				total_spent = payment_prices.sum()
				mean_price = payment_prices.mean()
				median_price = payment_prices.median()
				std_price = payment_prices.std()

				price_consistency = 1 - (abs(mean_price - median_price) / mean_price) if mean_price > 0 else 1
				expensive_ratio = len(payment_prices[payment_prices > 3000]) / payment_count
				cheap_ratio = len(payment_prices[payment_prices < 800]) / payment_count
				max_price = payment_prices.max()
				price_stability = 1 - (std_price / mean_price) if mean_price > 0 else 1

				payment_features = {
					'payment_count': payment_count,
					'total_spent': total_spent,
					'mean_price': mean_price,
					'median_price': median_price,
					'std_price': std_price,
					'max_price': max_price,
					'price_consistency': price_consistency,
					'expensive_ratio': expensive_ratio,
					'cheap_ratio': cheap_ratio,
					'price_stability': price_stability,
					'financial_activity': min(1.0, payment_count / 15.0),
					'min_price': payment_prices.min(),
				}
			else:
				payment_features = {
					'payment_count': 0, 'total_spent': 0, 'mean_price': 0,
					'median_price': 0, 'std_price': 0, 'max_price': 0,
					'price_consistency': 1, 'expensive_ratio': 0,
					'cheap_ratio': 0, 'price_stability': 1, 'financial_activity': 0,
					'min_price': 0,
				}

			# ============================================================
			# 3. ФИНАНСОВЫЕ ПРИЗНАКИ - ПРОДВИНУТЫЕ (15) ← НОВОЕ!
			# ============================================================
			advanced_payment_features = {}
			if len(user_payments) > 0:
				prices = user_payments['price'].values

				# Перцентили распределения платежей
				advanced_payment_features['price_p25'] = np.percentile(prices, 25)
				advanced_payment_features['price_p75'] = np.percentile(prices, 75)
				advanced_payment_features['price_p90'] = np.percentile(prices, 90)
				advanced_payment_features['price_p95'] = np.percentile(prices, 95)

				# IQR (межквартильный размах)
				advanced_payment_features['price_iqr'] = np.percentile(prices, 75) - np.percentile(prices, 25)

				# Коэффициент вариации
				advanced_payment_features['price_cv'] = (std_price / mean_price) if mean_price > 0 else 0

				# Асимметрия (skewness) и эксцесс (kurtosis)
				advanced_payment_features['price_skewness'] = stats.skew(prices) if len(prices) > 2 else 0
				advanced_payment_features['price_kurtosis'] = stats.kurtosis(prices) if len(prices) > 3 else 0

				# Тренд размера покупок (растет/падает)
				if 'timestamp' in user_payments.columns and len(user_payments) >= 3:
					sorted_payments = user_payments.sort_values('timestamp')
					first_half = sorted_payments['price'].iloc[:len(sorted_payments) // 2].mean()
					second_half = sorted_payments['price'].iloc[len(sorted_payments) // 2:].mean()
					advanced_payment_features['spending_trend'] = (
																			  second_half - first_half) / first_half if first_half > 0 else 0
				else:
					advanced_payment_features['spending_trend'] = 0

				# Регулярность платежей (стандартное отклонение временных интервалов)
				if 'timestamp' in user_payments.columns and len(user_payments) >= 2:
					sorted_payments = user_payments.sort_values('timestamp')
					time_diffs = sorted_payments['timestamp'].diff().dt.days.dropna()
					advanced_payment_features['payment_regularity'] = time_diffs.std() if len(time_diffs) > 1 else 0
					advanced_payment_features['avg_days_between_payments'] = time_diffs.mean() if len(
						time_diffs) > 0 else 0
				else:
					advanced_payment_features['payment_regularity'] = 0
					advanced_payment_features['avg_days_between_payments'] = 0

				# Концентрация трат (сколько % от общей суммы составляют топ-3 покупки)
				top3_sum = sorted(prices, reverse=True)[:3]
				advanced_payment_features['spending_concentration'] = sum(
					top3_sum) / total_spent if total_spent > 0 else 0

				# Частота крупных покупок (>5000)
				advanced_payment_features['large_purchase_freq'] = len(prices[prices > 5000]) / len(prices)

				# Частота мелких покупок (<500)
				advanced_payment_features['small_purchase_freq'] = len(prices[prices < 500]) / len(prices)
			else:
				advanced_payment_features = {
					'price_p25': 0, 'price_p75': 0, 'price_p90': 0, 'price_p95': 0,
					'price_iqr': 0, 'price_cv': 0, 'price_skewness': 0, 'price_kurtosis': 0,
					'spending_trend': 0, 'payment_regularity': 0, 'avg_days_between_payments': 0,
					'spending_concentration': 0, 'large_purchase_freq': 0, 'small_purchase_freq': 0
				}

			# ============================================================
			# 4. ПОВЕДЕНЧЕСКИЕ ПАТТЕРНЫ (12)
			# ============================================================
			behavioral_features = {
				'total_actions': total_actions,
				'action_diversity': len(categories),
				'view_ratio': safe_get(action_types, 'view', 0) / total_actions,
				'click_ratio': safe_get(action_types, 'click', 0) / total_actions,
				'clickout_ratio': safe_get(action_types, 'clickout', 0) / total_actions,
				'u2i_ratio': safe_get(categories, 'u2i', 0) / total_actions,
				'search_ratio': safe_get(categories, 'search', 0) / total_actions,
				'catalog_ratio': safe_get(categories, 'catalog', 0) / total_actions,
				'engagement_level': min(1.0, total_actions / 100.0),
				'exploration_score': safe_ratio(safe_get(categories, 'search'), safe_get(categories, 'u2i', 1)),
				'impulse_score': safe_ratio(safe_get(categories, 'u2i'), safe_get(categories, 'search', 1)),
				'conversion_rate': safe_get(action_types, 'clickout', 0) / max(1, safe_get(action_types, 'view', 1)),
			}

			# ============================================================
			# 5. ВРЕМЕННЫЕ ПАТТЕРНЫ (25) ← НОВОЕ!
			# ============================================================
			temporal_features = {}
			if 'timestamp' in user_market.columns:
				user_market['hour'] = user_market['timestamp'].dt.hour
				user_market['dayofweek'] = user_market['timestamp'].dt.dayofweek
				user_market['day'] = user_market['timestamp'].dt.day

				# Активность по дням недели (7 признаков)
				for day in range(7):
					day_count = len(user_market[user_market['dayofweek'] == day])
					temporal_features[f'activity_dow_{day}'] = day_count / total_actions if total_actions > 0 else 0

				# Активность по периодам дня (4 признака)
				temporal_features['activity_night'] = len(
					user_market[(user_market['hour'] >= 0) & (user_market['hour'] < 6)]) / total_actions
				temporal_features['activity_morning'] = len(
					user_market[(user_market['hour'] >= 6) & (user_market['hour'] < 12)]) / total_actions
				temporal_features['activity_afternoon'] = len(
					user_market[(user_market['hour'] >= 12) & (user_market['hour'] < 18)]) / total_actions
				temporal_features['activity_evening'] = len(
					user_market[(user_market['hour'] >= 18) & (user_market['hour'] < 24)]) / total_actions

				# Пиковое время активности
				hour_dist = user_market['hour'].value_counts()
				temporal_features['peak_hour'] = hour_dist.idxmax() if len(hour_dist) > 0 else 12
				temporal_features['peak_hour_ratio'] = hour_dist.max() / total_actions if len(hour_dist) > 0 else 0

				# Количество уникальных дней активности
				unique_days = user_market['timestamp'].dt.date.nunique()
				temporal_features['active_days_count'] = unique_days
				temporal_features['avg_actions_per_day'] = total_actions / unique_days if unique_days > 0 else 0

				# Давность последней активности (в днях)
				last_activity = user_market['timestamp'].max()
				days_since_last = (market_df['timestamp'].max() - last_activity).days
				temporal_features['days_since_last_activity'] = days_since_last
				temporal_features['is_recent_user'] = 1 if days_since_last <= 7 else 0

				# Регулярность активности
				sorted_market = user_market.sort_values('timestamp')
				if len(sorted_market) >= 2:
					time_gaps = sorted_market['timestamp'].diff().dt.days.dropna()
					temporal_features['avg_gap_days'] = time_gaps.mean() if len(time_gaps) > 0 else 0
					temporal_features['std_gap_days'] = time_gaps.std() if len(time_gaps) > 1 else 0
				else:
					temporal_features['avg_gap_days'] = 0
					temporal_features['std_gap_days'] = 0

				# Сезонность (по месяцам)
				month_dist = user_market['timestamp'].dt.month.value_counts()
				temporal_features['most_active_month'] = month_dist.idxmax() if len(month_dist) > 0 else 1
				temporal_features['month_concentration'] = month_dist.max() / total_actions if len(
					month_dist) > 0 else 0

				# Выходные vs будни
				temporal_features['weekend_ratio'] = len(
					user_market[user_market['dayofweek'].isin([5, 6])]) / total_actions
			else:
				# Заполнение нулями, если нет timestamp
				for day in range(7):
					temporal_features[f'activity_dow_{day}'] = 0
				temporal_features.update({
					'activity_night': 0, 'activity_morning': 0, 'activity_afternoon': 0,
					'activity_evening': 0, 'peak_hour': 12, 'peak_hour_ratio': 0,
					'active_days_count': 0, 'avg_actions_per_day': 0,
					'days_since_last_activity': 0, 'is_recent_user': 0,
					'avg_gap_days': 0, 'std_gap_days': 0,
					'most_active_month': 1, 'month_concentration': 0, 'weekend_ratio': 0
				})

			# ============================================================
			# 6. ПРОДУКТОВЫЕ ПРЕДПОЧТЕНИЯ - РАСШИРЕННЫЕ (18) ← РАСШИРЕНО!
			# ============================================================
			product_features = {}

			# Частота по каждой категории (6 признаков)
			for category in self.product_categories:
				cat_ratio = safe_get(product_categories, category, 0) / total_actions if total_actions > 0 else 0
				product_features[f'product_{category}_ratio'] = cat_ratio

			# Топ-3 категории (3 признака)
			if len(product_categories) > 0:
				top_categories = product_categories.head(3)
				for i, (cat, count) in enumerate(top_categories.items()):
					product_features[f'top_category_{i + 1}_ratio'] = count / total_actions
				# Дополняем до 3, если меньше
				for i in range(len(top_categories), 3):
					product_features[f'top_category_{i + 1}_ratio'] = 0
			else:
				for i in range(3):
					product_features[f'top_category_{i + 1}_ratio'] = 0

			# Антитоп-3 категории (самые непопулярные среди просмотренных) (3 признака)
			if len(product_categories) >= 3:
				bottom_categories = product_categories.tail(3)
				for i, (cat, count) in enumerate(bottom_categories.items()):
					product_features[f'bottom_category_{i + 1}_ratio'] = count / total_actions
			else:
				for i in range(3):
					product_features[f'bottom_category_{i + 1}_ratio'] = 0

			# Diversity score (энтропия распределения продуктов) (1 признак)
			if len(product_categories) > 0:
				probs = product_categories / product_categories.sum()
				product_features['product_diversity'] = -sum(probs * np.log(probs + 1e-10))
			else:
				product_features['product_diversity'] = 0

			# Concentration index (насколько сосредоточен на одной категории) (1 признак)
			if len(product_categories) > 0:
				product_features['product_concentration'] = product_categories.max() / product_categories.sum()
			else:
				product_features['product_concentration'] = 0

			# Количество уникальных категорий (1 признак)
			product_features['unique_product_categories'] = len(product_categories)

			# Доля основной категории (1 признак)
			if len(product_categories) > 0:
				product_features['main_category_dominance'] = product_categories.iloc[0] / total_actions
			else:
				product_features['main_category_dominance'] = 0

			# Склонность к кредитным продуктам (1 признак)
			credit_products = ['credit_cards', 'mortgage', 'auto_loans']
			credit_count = sum(safe_get(product_categories, cat, 0) for cat in credit_products)
			product_features['credit_products_affinity'] = credit_count / total_actions if total_actions > 0 else 0

			# Склонность к сберегательным продуктам (1 признак)
			saving_products = ['deposits', 'investments']
			saving_count = sum(safe_get(product_categories, cat, 0) for cat in saving_products)
			product_features['saving_products_affinity'] = saving_count / total_actions if total_actions > 0 else 0

			# ============================================================
			# 7. ЧАСТОТНЫЕ ХАРАКТЕРИСТИКИ (10) ← НОВОЕ!
			# ============================================================
			frequency_features = {}

			# Энтропия действий
			if len(action_types) > 0:
				action_probs = action_types / action_types.sum()
				frequency_features['action_entropy'] = -sum(action_probs * np.log(action_probs + 1e-10))
			else:
				frequency_features['action_entropy'] = 0

			# Энтропия доменов
			if len(categories) > 0:
				domain_probs = categories / categories.sum()
				frequency_features['domain_entropy'] = -sum(domain_probs * np.log(domain_probs + 1e-10))
			else:
				frequency_features['domain_entropy'] = 0

			# Частота смены категорий (переключения между категориями)
			if 'product_category' in user_market.columns and len(user_market) > 1:
				sorted_market = user_market.sort_values(
					'timestamp') if 'timestamp' in user_market.columns else user_market
				category_switches = (
							sorted_market['product_category'] != sorted_market['product_category'].shift()).sum()
				frequency_features['category_switch_rate'] = category_switches / len(user_market)
			else:
				frequency_features['category_switch_rate'] = 0

			# Частота смены типа действия
			if len(user_market) > 1:
				sorted_market = user_market.sort_values(
					'timestamp') if 'timestamp' in user_market.columns else user_market
				action_switches = (sorted_market['action_type'] != sorted_market['action_type'].shift()).sum()
				frequency_features['action_switch_rate'] = action_switches / len(user_market)
			else:
				frequency_features['action_switch_rate'] = 0

			# Bounce rate (быстрые выходы - одиночные действия)
			if 'timestamp' in user_market.columns:
				actions_per_day = user_market.groupby(user_market['timestamp'].dt.date).size()
				single_action_days = (actions_per_day == 1).sum()
				frequency_features['bounce_rate'] = single_action_days / len(actions_per_day) if len(
					actions_per_day) > 0 else 0
			else:
				frequency_features['bounce_rate'] = 0

			# Глубина просмотра (среднее количество действий за сессию)
			if 'timestamp' in user_market.columns:
				actions_per_day = user_market.groupby(user_market['timestamp'].dt.date).size()
				frequency_features['avg_session_depth'] = actions_per_day.mean() if len(actions_per_day) > 0 else 0
			else:
				frequency_features['avg_session_depth'] = 0

			# Повторные визиты (количество дней с активностью > 1 раз)
			if 'timestamp' in user_market.columns:
				daily_counts = user_market.groupby(user_market['timestamp'].dt.date).size()
				frequency_features['repeat_visit_days'] = (daily_counts > 1).sum()
				frequency_features['repeat_visit_ratio'] = (daily_counts > 1).sum() / len(daily_counts) if len(
					daily_counts) > 0 else 0
			else:
				frequency_features['repeat_visit_days'] = 0
				frequency_features['repeat_visit_ratio'] = 0

			# Интенсивность активности (std количества действий по дням)
			if 'timestamp' in user_market.columns:
				daily_counts = user_market.groupby(user_market['timestamp'].dt.date).size()
				frequency_features['activity_intensity_std'] = daily_counts.std() if len(daily_counts) > 1 else 0
			else:
				frequency_features['activity_intensity_std'] = 0

			# ============================================================
			# 8. ДОПОЛНИТЕЛЬНЫЕ ПРИЗНАКИ (5)
			# ============================================================
			additional_features = {
				'unique_action_types': len(action_types),
				'unique_categories': len(categories),
				'action_per_category': total_actions / len(categories) if len(categories) > 0 else 0,
				'has_payments': 1 if len(user_payments) > 0 else 0,
				'payment_to_action_ratio': len(user_payments) / total_actions if total_actions > 0 else 0,
			}

			# ============================================================
			# КОМБИНИРУЕМ ВСЕ ХАРАКТЕРИСТИКИ
			# ============================================================
			vector = {
				**demographic_features,
				**payment_features,
				**advanced_payment_features,
				**behavioral_features,
				**temporal_features,
				**product_features,
				**frequency_features,
				**additional_features
			}

			# Информация для интерпретации
			info = {
				'user_id': user_id,
				'age_group': age_group,
				'total_actions': total_actions,
				'payment_count': payment_features.get('payment_count', 0),
				'total_spent': payment_features.get('total_spent', 0),
				'behavior_type': self._classify_behavior(categories, action_types, total_actions),
				'financial_profile': self._classify_financial(payment_features.get('mean_price', 0),
															  payment_features.get('payment_count', 0)),
				'top_categories': list(product_categories.head(3).index) if len(product_categories) > 0 else []
			}

			return vector, info

		except Exception as e:
			print(f"❌ Ошибка создания вектора для пользователя {user_id}: {e}")
			return None, None

	def _classify_behavior(self, categories, action_types, total_actions):
		"""Классификация поведения"""
		if total_actions == 0:
			return "Неактивный"

		search_ratio = categories.get('search', 0) / total_actions
		u2i_ratio = categories.get('u2i', 0) / total_actions

		if u2i_ratio > 0.4:
			return "Импульсивный"
		elif search_ratio > 0.3:
			return "Исследователь"
		elif total_actions > 40:
			return "Активный"
		else:
			return "Умеренный"

	def _classify_financial(self, avg_transaction, payment_count):
		"""Классификация финансового профиля"""
		if payment_count == 0:
			return "Без транзакций"
		elif avg_transaction > 4000:
			return "Высокий доход"
		elif avg_transaction > 1500:
			return "Средний доход"
		else:
			return "Экономный"