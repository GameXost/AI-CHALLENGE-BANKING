import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

print("🎯 УНИВЕРСАЛЬНАЯ СИСТЕМА РЕКОМЕНДАЦИЙ ПСБ")
print("=" * 55)


class UniversalPSBRecommender:
    def __init__(self):
        # Полный список продуктов ПСБ с правильными названиями
        self.psb_products = {
            # Кредиты
            'credit_opk': "Кредит для работников предприятий ОПК и военнослужащих",
            'credit_any': "Кредит на любые цели",
            'credit_express': "Экспресс-кредит «Турбоденьги»",

            # Карты
            'debit_cashback': "Дебетовая карта «Твой кешбэк»",
            'card_resident': "Карта жителя",

            # Вклады
            'deposit_future': "Вклад «Ставка на будущее»",
            'deposit_precious': "Вклад «Драгоценный»",
            'deposit_strong': "Вклад «Сильная ставка»",
            'deposit_income': "Вклад «Мой доход»",
            'deposit_stable': "Вклад «Стабильный доход»",
            'deposit_savings': "Вклад «Моя копилка»",
            'deposit_flexible': "Вклад «Мои возможности»",
            'deposit_yuan': "Вклад «В юанях»",
            'deposit_social': "Вклад «Социальный вклад»",

            # Накопительные счета
            'savings_focus': "Накопительный счет «Акцент на процент»",
            'savings_reserve': "Накопительный счет «Про запас»",
            'savings_keeper': "Накопительный счет «Хранитель»",

            # Премиум и инвестиции
            'premium_orange': "Orange Premium Club",
            'premium_private': "Private Banking",
            'investments': "Инвестиции",

            # Ипотека и страхование
            'mortgage': "Ипотека",
            'insurance': "Страхование",
            'cashback_partners': "Кешбэк и скидки от партнеров"
        }

    def load_data(self):
        """Загрузка всех данных"""
        print("1. 📊 ЗАГРУЗКА ДАННЫХ")
        print("-" * 25)

        users_df = pd.read_parquet("Dataset_case/users.pq")
        market_df = pd.read_parquet("Dataset_case/marketplace/events/01000.pq")
        payments_df = pd.read_parquet("Dataset_case/payments/events/01000.pq")
        retail_df = pd.read_parquet("Dataset_case/retail/events/01000.pq")

        print(f"✅ Пользователей: {len(users_df):,}")
        print(f"✅ Событий: {len(market_df):,}")

        return users_df, market_df, payments_df, retail_df

    def create_advanced_profiles(self, users_df, market_df, payments_df, retail_df, sample_size=500):
        """Создание продвинутых профилей"""
        print(f"\n2. 🎪 АНАЛИЗ ПОВЕДЕНИЯ")
        print("-" * 25)

        active_users = market_df['user_id'].unique()
        sample_users = pd.Series(active_users).sample(
            min(sample_size, len(active_users)), random_state=42)

        profiles = []

        for user_id in sample_users:
            profile = self._analyze_advanced_profile(
                user_id, users_df, market_df, payments_df, retail_df)
            if profile:
                profiles.append(profile)

        print(f"✅ Создано профилей: {len(profiles)}")
        return pd.DataFrame(profiles)

    def _analyze_advanced_profile(self, user_id, users_df, market_df, payments_df, retail_df):
        """Продвинутый анализ профиля"""
        try:
            user_demo = users_df[users_df['user_id'] == user_id].iloc[0]
            market_actions = market_df[market_df['user_id'] == user_id]
            payment_actions = payments_df[payments_df['user_id'] == user_id]
            retail_actions = retail_df[retail_df['user_id'] == user_id]

            if len(market_actions) == 0:
                return None

            # Анализ поведения
            total_actions = len(market_actions)
            action_types = market_actions['action_type'].value_counts()
            categories = market_actions['subdomain'].value_counts()

            # Сложные метрики
            view_ratio = action_types.get('view', 0) / total_actions
            click_ratio = action_types.get('click', 0) / total_actions
            research_ratio = categories.get('search', 0) / total_actions
            u2i_ratio = categories.get('u2i', 0) / total_actions

            # Финансовое поведение
            avg_transaction = payment_actions['price'].abs(
            ).mean() if len(payment_actions) > 0 else 0

            profile = {
                'user_id': user_id,
                'socdem_cluster': user_demo['socdem_cluster'],
                'region': user_demo['region'],

                # Поведенческие метрики
                'total_actions': total_actions,
                'view_ratio': view_ratio,
                'click_ratio': click_ratio,
                'research_ratio': research_ratio,
                'u2i_ratio': u2i_ratio,
                'avg_transaction': avg_transaction,

                # Демографические сегменты
                'is_student': user_demo['socdem_cluster'] in [0, 1, 2],
                'is_young': user_demo['socdem_cluster'] in [3, 4, 5],
                'is_young_family': user_demo['socdem_cluster'] in [6, 7, 8],
                'is_family': user_demo['socdem_cluster'] in [9, 10, 11],
                'is_mature': user_demo['socdem_cluster'] in [12, 13, 14],
                'is_senior': user_demo['socdem_cluster'] in [15, 16, 17],
                'is_affluent': user_demo['socdem_cluster'] in [18, 19, 20, 21],

                # Поведенческие типы
                'is_researcher': research_ratio > 0.4,
                'is_impulsive': u2i_ratio > 0.6,
                'is_active': total_actions > 30,
                'is_high_spender': avg_transaction > 5000,
                'is_metro': user_demo['region'] in [1, 2, 3]
            }

            return profile

        except:
            return None

    def ml_segmentation(self, profiles_df):
        """ML сегментация пользователей"""
        print(f"\n3. 🤖 ML СЕГМЕНТАЦИЯ")
        print("-" * 20)

        if len(profiles_df) == 0:
            return profiles_df

        # Признаки для кластеризации
        features = [
            'view_ratio', 'click_ratio', 'research_ratio', 'u2i_ratio',
            'avg_transaction', 'total_actions'
        ]

        X = profiles_df[features].fillna(0)

        # Масштабирование и кластеризация
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        profiles_df['cluster'] = clusters

        # Названия кластеров
        cluster_names = {
            0: "Исследователи",
            1: "Импульсивные покупатели",
            2: "Экономные плановики",
            3: "Премиальные клиенты",
            4: "Активные шопперы",
            5: "Новые пользователи"
        }

        profiles_df['segment_name'] = profiles_df['cluster'].map(cluster_names)

        print("📊 СЕГМЕНТЫ ПОЛЬЗОВАТЕЛЕЙ:")
        segment_stats = profiles_df['segment_name'].value_counts()
        for segment, count in segment_stats.items():
            print(f"   • {segment}: {count}")

        return profiles_df

    def generate_universal_recommendations(self, segmented_df):
        """Универсальная генерация рекомендаций"""
        print(f"\n4. 💡 ПЕРСОНАЛИЗИРОВАННЫЕ РЕКОМЕНДАЦИИ")
        print("-" * 40)

        recommendations = []

        for _, user in segmented_df.iterrows():
            user_recs = self._get_universal_recommendations(user)
            if user_recs:
                recommendations.append({
                    'user_id': user['user_id'],
                    'segment': user['segment_name'],
                    'demographics': self._get_demographic_group(user),
                    'recommendations': user_recs
                })

        print(f"✅ Сгенерировано рекомендаций: {len(recommendations)}")
        return recommendations

    def _get_universal_recommendations(self, user):
        """Универсальные рекомендации на основе всех факторов"""
        recs = []

        # 🔥 СЛОЖНАЯ ЛОГИКА С УЧЕТОМ ВСЕХ ФАКТОРОВ

        # МОЛОДЫЕ СТУДЕНТЫ
        if user['is_student'] and user['is_active']:
            recs.extend([
                {'product': 'debit_cashback',
                    'reason': 'Кешбэк за покупки для студентов'},
                {'product': 'credit_express', 'reason': 'Быстрый кредит для учебы'},
                {'product': 'savings_focus', 'reason': 'Накопления на будущее'}
            ])

        # МОЛОДЫЕ СЕМЬИ
        elif user['is_young_family'] and user['is_metro']:
            recs.extend([
                {'product': 'mortgage', 'reason': 'Ипотека для молодой семьи'},
                {'product': 'deposit_savings', 'reason': 'Накопления на детей'},
                {'product': 'insurance', 'reason': 'Страхование семьи'},
                {'product': 'card_resident', 'reason': 'Льготы для жителей'}
            ])

        # ПРЕМИАЛЬНЫЕ КЛИЕНТЫ
        elif user['is_affluent'] and user['is_high_spender']:
            recs.extend([
                {'product': 'premium_orange', 'reason': 'Премиальное обслуживание'},
                {'product': 'deposit_strong', 'reason': 'Максимальная ставка'},
                {'product': 'investments', 'reason': 'Инвестиционные решения'},
                {'product': 'premium_private', 'reason': 'Private Banking'}
            ])

        # АКТИВНЫЕ ИССЛЕДОВАТЕЛИ
        elif user['is_researcher'] and user['total_actions'] > 50:
            recs.extend([
                {'product': 'deposit_flexible', 'reason': 'Гибкие условия'},
                {'product': 'savings_keeper', 'reason': 'Индивидуальные правила'},
                {'product': 'credit_any', 'reason': 'Кредит для любых целей'}
            ])

        # ИМПУЛЬСИВНЫЕ ПОКУПАТЕЛИ
        elif user['is_impulsive'] and user['u2i_ratio'] > 0.7:
            recs.extend([
                {'product': 'cashback_partners', 'reason': 'Скидки у партнеров'},
                {'product': 'savings_reserve',
                    'reason': 'Защита от импульсивных трат'},
                {'product': 'debit_cashback', 'reason': 'Возврат средств'}
            ])

        # УНИВЕРСАЛЬНЫЕ РЕКОМЕНДАЦИИ ДЛЯ ВСЕХ
        base_recs = [
            {'product': 'deposit_income', 'reason': 'Стабильный доход'},
            {'product': 'debit_cashback', 'reason': 'Кешбэк за покупки'}
        ]

        # Добавляем базовые, если мало рекомендаций
        if len(recs) < 2:
            recs.extend(base_recs)

        # Убираем дубликаты
        seen = set()
        unique_recs = []
        for rec in recs:
            if rec['product'] not in seen:
                seen.add(rec['product'])
                unique_recs.append(rec)

        return unique_recs[:4]  # До 4 рекомендаций

    def _get_demographic_group(self, user):
        """Группа демографии"""
        if user['is_student']:
            return "Студент"
        elif user['is_young_family']:
            return "Молодая семья"
        elif user['is_affluent']:
            return "Премиум клиент"
        elif user['is_senior']:
            return "Пенсионер"
        else:
            return "Стандарт"

    def show_detailed_results(self, recommendations):
        """Детальные результаты"""
        print(f"\n5. 📊 РЕЗУЛЬТАТЫ СИСТЕМЫ")
        print("-" * 25)

        if not recommendations:
            print("❌ Нет рекомендаций")
            return

        # Статистика по продуктам
        product_stats = {}
        for rec in recommendations:
            for product_rec in rec['recommendations']:
                product = product_rec['product']
                product_stats[product] = product_stats.get(product, 0) + 1

        print("🏆 ТОП РЕКОМЕНДАЦИЙ:")
        for product, count in sorted(product_stats.items(), key=lambda x: x[1], reverse=True)[:8]:
            product_name = self.psb_products[product]
            print(f"   • {product_name}: {count}")

        # Примеры
        print(f"\n6. 🎯 ПРИМЕРЫ РЕКОМЕНДАЦИЙ")
        print("-" * 30)

        for rec in recommendations[:6]:
            print(f"\n👤 Клиент {str(rec['user_id'])[:8]}...")
            print(f"   📍 {rec['segment']} • {rec['demographics']}")
            for product_rec in rec['recommendations']:
                product_name = self.psb_products[product_rec['product']]
                print(f"   • {product_name}")
                print(f"     → {product_rec['reason']}")

# Запуск системы


def main():
    recommender = UniversalPSBRecommender()

    # 1. Загрузка данных
    users_df, market_df, payments_df, retail_df = recommender.load_data()

    # 2. Создание профилей
    profiles_df = recommender.create_advanced_profiles(
        users_df, market_df, payments_df, retail_df, 500)

    if len(profiles_df) == 0:
        print("❌ Не удалось создать профили")
        return

    # 3. ML сегментация
    segmented_df = recommender.ml_segmentation(profiles_df)

    # 4. Генерация рекомендаций
    recommendations = recommender.generate_universal_recommendations(
        segmented_df)

    # 5. Результаты
    recommender.show_detailed_results(recommendations)

    print(f"\n{'='*55}")
    print("✅ УНИВЕРСАЛЬНАЯ СИСТЕМА ПСБ ГОТОВА!")
    print("=" * 55)


if __name__ == "__main__":
    main()
