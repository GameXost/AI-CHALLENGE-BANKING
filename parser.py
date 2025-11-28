import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("УНИВЕРСАЛЬНАЯ ML СИСТЕМА РЕКОМЕНДАЦИЙ ДЛЯ БАНКА")
print("=" * 60)


class UniversalBankingRecommender:
    def __init__(self):
        self.products = {
            'credit_express': "Экспресс-кредит",
            'debit_cashback': "Дебетовая карта с кешбэком",
            'deposit_strong': "Вклад с высокой ставкой",
            'savings_account': "Накопительный счет",
            'premium_service': "Премиальное обслуживание",
            'investments': "Инвестиции",
            'mortgage': "Ипотека",
            'insurance': "Страхование"
        }

        self.ml_model = None
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        self.feature_columns = []

    def load_data(self):
        print("ЗАГРУЗКА ДАННЫХ...")
        try:
            data_path = Path("Dataset_case")
            if not data_path.exists():
                print("Папка не найдена. Создаем реалистичные демо-данные...")
                return self._create_smart_demo_data()

            users_df = self._safe_load_parquet(data_path / "users.pq", "users")
            if users_df is None:
                users_df = self._create_demo_users(1000)

            market_df = self._safe_load_sample(data_path, "marketplace", 20000)
            payments_df = self._safe_load_sample(data_path, "payments", 10000)
            retail_df = self._safe_load_sample(data_path, "retail", 5000)

            return users_df, market_df, payments_df, retail_df

        except Exception as e:
            print(f"Ошибка загрузки: {e}")
            return self._create_smart_demo_data()

    def _safe_load_parquet(self, path, name):
        try:
            if path.exists():
                df = pd.read_parquet(path)
                print(f"{name}: {len(df)} записей")
                return df
            else:
                print(f"{name}: файл не найден")
                return None
        except Exception as e:
            print(f"Ошибка загрузки {name}: {e}")
            return None

    def _safe_load_sample(self, data_path, event_type, sample_size):
        try:
            event_files = list(data_path.glob(f"{event_type}/events/*.pq"))
            if event_files:
                df = pd.read_parquet(event_files[0])
                if len(df) > sample_size:
                    df = df.sample(sample_size, random_state=42)
                print(f"{event_type}: {len(df)} событий")
                return df
            else:
                print(f"{event_type}: файлы не найдены")
                return pd.DataFrame()
        except Exception as e:
            print(f"Ошибка {event_type}: {e}")
            return pd.DataFrame()

    def _create_demo_users(self, n_users):
        age_groups = ['18-25', '26-35', '36-45', '46-55', '55+']
        users_df = pd.DataFrame({
            'user_id': range(1, n_users + 1),
            'socdem_cluster': np.random.randint(0, 10, n_users),
            'region': np.random.randint(1, 50, n_users)
        })
        if 'age_group' not in users_df.columns:
            users_df['age_group'] = np.random.choice(
                age_groups, n_users, p=[0.2, 0.3, 0.25, 0.15, 0.1])
        return users_df

    def _create_smart_demo_data(self):
        print("Создание реалистичных демо-данных...")
        np.random.seed(42)

        users_df = self._create_demo_users(2000)

        market_data = []
        payments_data = []
        retail_data = []

        user_groups = {
            'young_active': users_df[users_df['age_group'].isin(['18-25', '26-35'])]['user_id'].sample(400),
            'family_oriented': users_df[users_df['age_group'].isin(['36-45'])]['user_id'].sample(300),
            'affluent': users_df.sample(200)['user_id'],
            'researchers': users_df.sample(300)['user_id'],
            'impulsive': users_df.sample(250)['user_id'],
            'conservative': users_df[users_df['age_group'].isin(['46-55', '55+'])]['user_id'].sample(200),
            'digital_natives': users_df[users_df['age_group'].isin(['18-25'])]['user_id'].sample(150),
            'business_owners': users_df.sample(100)['user_id']
        }

        for group_name, user_ids in user_groups.items():
            for user_id in user_ids:
                if group_name == 'young_active':
                    n_actions = np.random.randint(20, 80)
                    for _ in range(n_actions):
                        market_data.append({
                            'user_id': user_id,
                            'action_type': np.random.choice(['view', 'click', 'clickout'], p=[0.5, 0.3, 0.2]),
                            'subdomain': np.random.choice(['u2i', 'search', 'catalog'], p=[0.4, 0.3, 0.3])
                        })
                    n_payments = np.random.randint(5, 15)
                    for _ in range(n_payments):
                        payments_data.append({
                            'user_id': user_id,
                            'price': np.random.normal(2000, 800)
                        })

                elif group_name == 'family_oriented':
                    n_actions = np.random.randint(15, 50)
                    for _ in range(n_actions):
                        market_data.append({
                            'user_id': user_id,
                            'action_type': np.random.choice(['view', 'click', 'clickout'], p=[0.4, 0.3, 0.3]),
                            'subdomain': np.random.choice(['u2i', 'search', 'catalog'], p=[0.3, 0.5, 0.2])
                        })
                    n_payments = np.random.randint(8, 20)
                    for _ in range(n_payments):
                        payments_data.append({
                            'user_id': user_id,
                            'price': np.random.normal(4000, 1500)
                        })

                elif group_name == 'affluent':
                    n_actions = np.random.randint(10, 30)
                    for _ in range(n_actions):
                        market_data.append({
                            'user_id': user_id,
                            'action_type': np.random.choice(['view', 'click', 'clickout'], p=[0.6, 0.2, 0.2]),
                            'subdomain': np.random.choice(['u2i', 'search', 'catalog'], p=[0.2, 0.6, 0.2])
                        })
                    n_payments = np.random.randint(15, 40)
                    for _ in range(n_payments):
                        payments_data.append({
                            'user_id': user_id,
                            'price': np.random.normal(8000, 3000)
                        })

                elif group_name == 'researchers':
                    n_actions = np.random.randint(30, 100)
                    for _ in range(n_actions):
                        market_data.append({
                            'user_id': user_id,
                            'action_type': np.random.choice(['view', 'click', 'clickout'], p=[0.3, 0.4, 0.3]),
                            'subdomain': np.random.choice(['u2i', 'search', 'catalog'], p=[0.2, 0.7, 0.1])
                        })
                    n_payments = np.random.randint(3, 10)
                    for _ in range(n_payments):
                        payments_data.append({
                            'user_id': user_id,
                            'price': np.random.normal(3000, 1000)
                        })

                elif group_name == 'impulsive':
                    n_actions = np.random.randint(25, 70)
                    for _ in range(n_actions):
                        market_data.append({
                            'user_id': user_id,
                            'action_type': np.random.choice(['view', 'click', 'clickout'], p=[0.2, 0.3, 0.5]),
                            'subdomain': np.random.choice(['u2i', 'search', 'catalog'], p=[0.7, 0.2, 0.1])
                        })
                    n_payments = np.random.randint(10, 25)
                    for _ in range(n_payments):
                        payments_data.append({
                            'user_id': user_id,
                            'price': np.random.normal(2500, 1200)
                        })

                elif group_name == 'conservative':
                    n_actions = np.random.randint(5, 20)
                    for _ in range(n_actions):
                        market_data.append({
                            'user_id': user_id,
                            'action_type': np.random.choice(['view', 'click', 'clickout'], p=[0.7, 0.2, 0.1]),
                            'subdomain': np.random.choice(['u2i', 'search', 'catalog'], p=[0.2, 0.3, 0.5])
                        })
                    n_payments = np.random.randint(2, 8)
                    for _ in range(n_payments):
                        payments_data.append({
                            'user_id': user_id,
                            'price': np.random.normal(5000, 2000)
                        })

                elif group_name == 'digital_natives':
                    n_actions = np.random.randint(40, 120)
                    for _ in range(n_actions):
                        market_data.append({
                            'user_id': user_id,
                            'action_type': np.random.choice(['view', 'click', 'clickout'], p=[0.4, 0.4, 0.2]),
                            'subdomain': np.random.choice(['u2i', 'search', 'catalog'], p=[0.5, 0.3, 0.2])
                        })
                    n_payments = np.random.randint(8, 20)
                    for _ in range(n_payments):
                        payments_data.append({
                            'user_id': user_id,
                            'price': np.random.normal(1500, 600)
                        })

                elif group_name == 'business_owners':
                    n_actions = np.random.randint(20, 60)
                    for _ in range(n_actions):
                        market_data.append({
                            'user_id': user_id,
                            'action_type': np.random.choice(['view', 'click', 'clickout'], p=[0.5, 0.3, 0.2]),
                            'subdomain': np.random.choice(['u2i', 'search', 'catalog'], p=[0.3, 0.5, 0.2])
                        })
                    n_payments = np.random.randint(20, 50)
                    for _ in range(n_payments):
                        payments_data.append({
                            'user_id': user_id,
                            'price': np.random.normal(10000, 5000)
                        })

        market_df = pd.DataFrame(market_data)
        payments_df = pd.DataFrame(payments_data)
        retail_df = pd.DataFrame(retail_data)

        print("Созданы реалистичные демо-данные")
        return users_df, market_df, payments_df, retail_df

    def create_feature_vectors(self, users_df, market_df, payments_df, retail_df, sample_size=200):
        print("\n1. СОЗДАНИЕ ВЕКТОРОВ ДЛЯ ML")
        print("-" * 35)

        if len(market_df) == 0:
            print("Нет данных маркетплейса")
            return pd.DataFrame(), pd.DataFrame()

        active_users = market_df['user_id'].unique()
        sample_users = np.random.choice(active_users, min(
            sample_size, len(active_users)), replace=False)

        print(f"Обрабатывается {len(sample_users)} пользователей...")

        vectors = []
        user_info = []

        for i, user_id in enumerate(sample_users):
            try:
                vector, info = self._compute_user_vector(
                    user_id, users_df, market_df, payments_df)
                if vector is not None:
                    vectors.append(vector)
                    user_info.append(info)

                if (i + 1) % 50 == 0:
                    print(f"   Обработано: {i + 1}/{len(sample_users)}")

            except Exception as e:
                continue

        if not vectors:
            print("Не удалось создать векторы")
            return pd.DataFrame(), pd.DataFrame()

        feature_matrix = pd.DataFrame(vectors)
        info_df = pd.DataFrame(user_info)

        self.feature_columns = [
            col for col in feature_matrix.columns if col != 'user_id']

        print(f"Создано векторов: {len(vectors)}")
        print(f"Признаков на пользователя: {len(self.feature_columns)}")

        return feature_matrix, info_df

    def _compute_user_vector(self, user_id, users_df, market_df, payments_df):
        try:
            user_data = users_df[users_df['user_id'] == user_id]
            if len(user_data) == 0:
                return None, None
            user_data = user_data.iloc[0]

            user_market = market_df[market_df['user_id'] == user_id]
            user_payments = payments_df[payments_df['user_id'] == user_id] if len(
                payments_df) > 0 else pd.DataFrame()

            if len(user_market) == 0:
                return None, None

            total_actions = len(user_market)
            action_types = user_market['action_type'].value_counts()
            categories = user_market['subdomain'].value_counts()

            def safe_ratio(numerator, denominator):
                return numerator / denominator if denominator > 0 else 0.0

            def safe_get(series, key, default=0):
                return series.get(key, default) if len(series) > 0 else default

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

            vector = {
                'user_id': user_id,
                'socdem_cluster': user_data['socdem_cluster'],
                'is_young': 1 if age_group in ['18-25', '26-35'] else 0,
                'is_family': 1 if age_group in ['36-45'] else 0,
                'is_mature': 1 if age_group in ['46-55', '55+'] else 0,
                'total_actions': total_actions,
                'action_diversity': len(categories),
                'view_ratio': safe_ratio(safe_get(action_types, 'view'), total_actions),
                'click_ratio': safe_ratio(safe_get(action_types, 'click'), total_actions),
                'clickout_ratio': safe_ratio(safe_get(action_types, 'clickout'), total_actions),
                'u2i_ratio': safe_ratio(safe_get(categories, 'u2i'), total_actions),
                'search_ratio': safe_ratio(safe_get(categories, 'search'), total_actions),
                'catalog_ratio': safe_ratio(safe_get(categories, 'catalog'), total_actions),
                'has_payments': 1 if len(user_payments) > 0 else 0,
                'payment_count': len(user_payments),
                'avg_transaction': user_payments['price'].mean() if len(user_payments) > 0 else 0,
                'total_spent': user_payments['price'].sum() if len(user_payments) > 0 else 0,
                'financial_activity': min(1.0, len(user_payments) / 30.0),
                'engagement_level': min(1.0, total_actions / 150.0),
                'exploration_score': safe_ratio(safe_get(categories, 'search'), safe_get(categories, 'u2i', 1)),
                'impulse_score': safe_ratio(safe_get(categories, 'u2i'), safe_get(categories, 'search', 1))
            }

            info = {
                'user_id': user_id,
                'age_group': age_group,
                'behavior_type': self._classify_behavior(categories, action_types, total_actions),
                'financial_profile': self._classify_financial(user_payments, vector['avg_transaction'])
            }

            return vector, info

        except Exception as e:
            return None, None

    def _classify_behavior(self, categories, action_types, total_actions):
        if total_actions == 0:
            return "Неактивный"

        search_ratio = categories.get('search', 0) / total_actions
        u2i_ratio = categories.get('u2i', 0) / total_actions

        if u2i_ratio > 0.5:
            return "Импульсивный"
        elif search_ratio > 0.4:
            return "Исследователь"
        elif total_actions > 60:
            return "Активный"
        else:
            return "Умеренный"

    def _classify_financial(self, payments, avg_transaction):
        if len(payments) == 0:
            return "Без транзакций"
        elif avg_transaction > 6000:
            return "Высокий доход"
        elif avg_transaction > 2500:
            return "Средний доход"
        else:
            return "Экономный"

    def train_single_ml_model(self, feature_matrix, info_df):
        print(f"\n2. ОБУЧЕНИЕ УНИВЕРСАЛЬНОЙ ML МОДЕЛИ")
        print("-" * 40)

        if len(feature_matrix) < 30:
            print("Недостаточно данных для обучения")
            return info_df

        X = feature_matrix[self.feature_columns].fillna(0)

        X_scaled = self.scaler.fit_transform(X)

        try:
            clusters = self.kmeans.fit_predict(X_scaled)
            info_df['ml_cluster'] = clusters
            print(
                f"K-Means: {len(np.unique(clusters))} кластеров пользователей")
        except Exception as e:
            print(f"Ошибка кластеризации: {e}")
            info_df['ml_cluster'] = 0

        print("Создание целевой переменной...")
        y = self._create_user_category(feature_matrix, info_df)

        print("Обучение единой Random Forest модели...")

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42, stratify=y
            )

            self.ml_model = RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=3,
                class_weight='balanced',
                random_state=42
            )

            self.ml_model.fit(X_train, y_train)

            train_score = self.ml_model.score(X_train, y_train)
            test_score = self.ml_model.score(X_test, y_test)

            print(f"Универсальная ML модель обучена:")
            print(f"   Размер выборки: {len(X)} пользователей")
            print(f"   Количество классов: {len(np.unique(y))}")
            print(f"   Train accuracy: {train_score:.3f}")
            print(f"   Test accuracy: {test_score:.3f}")

            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.ml_model.feature_importances_
            }).sort_values('importance', ascending=False)

            print(f"   Топ-5 важных признаков:")
            for i, row in feature_importance.head().iterrows():
                print(f"      {row['feature']}: {row['importance']:.3f}")

        except Exception as e:
            print(f"Ошибка обучения модели: {e}")
            self.ml_model = None

        return info_df

    def _create_user_category(self, feature_matrix, info_df):
        categories = []

        for i, row in feature_matrix.iterrows():
            if row['is_young'] == 1 and row['engagement_level'] > 0.3:
                category = "Молодой_активный"
            elif row['is_family'] == 1 and row['financial_activity'] > 0.2:
                category = "Семейный_финансовый"
            elif row['is_mature'] == 1 and row['search_ratio'] > 0.3:
                category = "Зрелый_исследователь"
            elif row['impulse_score'] > 0.4:
                category = "Импульсивный_покупатель"
            elif row['exploration_score'] > 0.5:
                category = "Любознательный_исследователь"
            elif row['financial_activity'] > 0.4:
                category = "Финансово_активный"
            elif row['engagement_level'] > 0.5:
                category = "Высоко_вовлеченный"
            else:
                category = "Стандартный_пользователь"

            categories.append(category)

        category_counts = pd.Series(categories).value_counts()
        print("Распределение категорий пользователей:")
        for category, count in category_counts.items():
            print(
                f"   {category}: {count} пользователей ({count/len(categories):.1%})")

        return categories

    def predict_with_single_model(self, feature_matrix, info_df):
        print(f"\n3. ML ПРЕДСКАЗАНИЕ (ОДНА МОДЕЛЬ)")
        print("-" * 35)

        if self.ml_model is None:
            print("ML модель не обучена")
            return []

        X = feature_matrix[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)

        results = []

        user_categories = self.ml_model.predict(X_scaled)
        user_probabilities = self.ml_model.predict_proba(X_scaled)

        for i in range(len(X_scaled)):
            user_category = user_categories[i]
            category_prob = np.max(user_probabilities[i])

            recommended_products = self._map_category_to_products(
                user_category)

            user_data = info_df.iloc[i]

            results.append({
                'user_id': user_data['user_id'],
                'demographics': f"{user_data['age_group']}",
                'profile': f"{user_data['behavior_type']}, {user_data['financial_profile']}",
                'ml_cluster': user_data['ml_cluster'],
                'user_category': user_category,
                'category_confidence': category_prob,
                'recommended_products': recommended_products,
                'top_recommendation': recommended_products[0] if recommended_products else "Нет рекомендаций"
            })

        print(f"ML предсказание завершено для {len(results)} пользователей")
        return results

    def _map_category_to_products(self, user_category):
        category_mapping = {
            "Молодой_активный": [
                "Дебетовая карта с кешбэком",
                "Экспресс-кредит",
                "Накопительный счет"
            ],
            "Семейный_финансовый": [
                "Ипотека",
                "Страхование",
                "Вклад с высокой ставкой"
            ],
            "Зрелый_исследователь": [
                "Инвестиции",
                "Вклад с высокой ставкой",
                "Страхование"
            ],
            "Импульсивный_покупатель": [
                "Дебетовая карта с кешбэком",
                "Экспресс-кредит",
                "Накопительный счет"
            ],
            "Любознательный_исследователь": [
                "Инвестиции",
                "Вклад с высокой ставкой",
                "Премиальное обслуживание"
            ],
            "Финансово_активный": [
                "Премиальное обслуживание",
                "Инвестиции",
                "Вклад с высокой ставкой"
            ],
            "Высоко_вовлеченный": [
                "Премиальное обслуживание",
                "Дебетовая карта с кешбэком",
                "Инвестиции"
            ],
            "Стандартный_пользователь": [
                "Дебетовая карта с кешбэком",
                "Накопительный счет",
                "Страхование"
            ]
        }

        return category_mapping.get(user_category, ["Дебетовая карта с кешбэком", "Накопительный счет"])

    def generate_recommendations(self, ml_results):
        print(f"\n4. ГЕНЕРАЦИЯ РЕКОМЕНДАЦИЙ")
        print("-" * 25)

        recommendations = []

        for result in ml_results:
            recommendations.append({
                'user_id': result['user_id'],
                'demographics': result['demographics'],
                'profile': result['profile'],
                'ml_cluster': result['ml_cluster'],
                'user_category': result['user_category'],
                'confidence': result['category_confidence'],
                'best_product': result['top_recommendation'],
                'all_recommendations': result['recommended_products']
            })

        return recommendations

    def run_complete_system(self, sample_size=200):
        print("\n" + "=" * 50)
        print("ML СИСТЕМА РЕКОМЕНДАЦИЙ ЗАПУЩЕНА")
        print("=" * 50)

        try:
            users_df, market_df, payments_df, retail_df = self.load_data()

            feature_matrix, info_df = self.create_feature_vectors(
                users_df, market_df, payments_df, retail_df, sample_size
            )

            if len(feature_matrix) == 0:
                print("Не удалось создать векторы")
                return []

            info_df = self.train_single_ml_model(feature_matrix, info_df)

            if self.ml_model is None:
                print("Не удалось обучить ML модель")
                return []

            ml_results = self.predict_with_single_model(
                feature_matrix, info_df)

            recommendations = self.generate_recommendations(ml_results)

            self.show_ml_results(recommendations)

            return recommendations

        except Exception as e:
            print(f"Критическая ошибка: {e}")
            import traceback
            traceback.print_exc()
            return []

    def show_ml_results(self, recommendations):
        print(f"\n5. РЕЗУЛЬТАТЫ ML СИСТЕМЫ")
        print("-" * 30)

        if not recommendations:
            print("Нет рекомендаций для показа")
            return

        for i, rec in enumerate(recommendations[:5], 1):
            print(f"\n[{i}] ПОЛЬЗОВАТЕЛЬ: {rec['user_id']}")
            print(f"    Демография: {rec['demographics']}")
            print(f"    Профиль: {rec['profile']}")
            print(f"    ML кластер: {rec['ml_cluster']}")
            print(f"    Категория: {rec['user_category']}")
            print(f"    Уверенность: {rec['confidence']:.1%}")
            print(f"    РЕКОМЕНДАЦИЯ: {rec['best_product']}")
            print(
                f"    Все рекомендации: {', '.join(rec['all_recommendations'])}")
            print("    " + "=" * 45)


def main():
    recommender = UniversalBankingRecommender()
    results = recommender.run_complete_system(200)

    print(f"\nML СИСТЕМА УСПЕШНО ЗАВЕРШИЛА РАБОТУ")
    print(f"Обработано пользователей: {len(results)}")
    print(f"Использована 1 универсальная ML модель")
    print(f"Алгоритмы: Random Forest + K-Means")
    print(f"Простая и эффективная архитектура!")


if __name__ == "__main__":
    main()
