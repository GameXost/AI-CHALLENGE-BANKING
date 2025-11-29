import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


class DataLoader:
    def load_data(self):
        print("ЗАГРУЗКА ДАННЫХ...")
        try:
            data_path = Path("Dataset_case")
            if not data_path.exists():
                print("Папка не найдена. Создаем демо-данные...")
                return self._create_demo_data()

            users_df = self._safe_load_parquet(data_path / "users.pq", "users")
            if users_df is None:
                users_df = self._create_demo_users(1000)

            market_df = self._safe_load_sample(data_path, "marketplace", 20000)
            payments_df = self._safe_load_sample(data_path, "payments", 10000)
            retail_df = self._safe_load_sample(data_path, "retail", 5000)

            return users_df, market_df, payments_df, retail_df

        except Exception as e:
            print(f"Ошибка загрузки: {e}")
            return self._create_demo_data()

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

    def _create_demo_data(self):
        print("Создание реалистичных демо-данных...")
        np.random.seed(42)

        users_df = self._create_demo_users(500)

        market_data = []
        payments_data = []
        retail_data = []

        dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
        product_categories = ['credit_cards', 'deposits',
                              'investments', 'insurance', 'mortgage', 'auto_loans']

        user_groups = {
            'young_active': users_df[users_df['age_group'].isin(['18-25', '26-35'])]['user_id'].sample(80),
            'family_oriented': users_df[users_df['age_group'].isin(['36-45'])]['user_id'].sample(60),
            'affluent': users_df.sample(40)['user_id'],
            'researchers': users_df.sample(60)['user_id'],
            'impulsive': users_df.sample(50)['user_id'],
            'conservative': users_df[users_df['age_group'].isin(['46-55', '55+'])]['user_id'].sample(40),
            'digital_natives': users_df[users_df['age_group'].isin(['18-25'])]['user_id'].sample(30),
            'business_owners': users_df.sample(20)['user_id']
        }

        for group_name, user_ids in user_groups.items():
            for user_id in user_ids:
                if group_name == 'young_active':
                    n_actions = np.random.randint(20, 80)
                    for i in range(n_actions):
                        market_data.append({
                            'user_id': user_id,
                            'action_type': np.random.choice(['view', 'click', 'clickout'], p=[0.5, 0.3, 0.2]),
                            'subdomain': np.random.choice(['u2i', 'search', 'catalog'], p=[0.4, 0.3, 0.3]),
                            'product_category': np.random.choice(product_categories, p=[0.3, 0.2, 0.1, 0.1, 0.2, 0.1]),
                            'timestamp': np.random.choice(dates)
                        })
                    n_payments = np.random.randint(5, 15)
                    for i in range(n_payments):
                        payments_data.append({
                            'user_id': user_id,
                            'price': np.random.normal(2000, 800),
                            'product_category': np.random.choice(product_categories, p=[0.3, 0.2, 0.1, 0.1, 0.2, 0.1]),
                            'timestamp': np.random.choice(dates)
                        })

                elif group_name == 'family_oriented':
                    n_actions = np.random.randint(15, 50)
                    for i in range(n_actions):
                        market_data.append({
                            'user_id': user_id,
                            'action_type': np.random.choice(['view', 'click', 'clickout'], p=[0.4, 0.3, 0.3]),
                            'subdomain': np.random.choice(['u2i', 'search', 'catalog'], p=[0.3, 0.5, 0.2]),
                            'product_category': np.random.choice(product_categories, p=[0.1, 0.3, 0.1, 0.2, 0.2, 0.1]),
                            'timestamp': np.random.choice(dates)
                        })
                    n_payments = np.random.randint(8, 20)
                    for i in range(n_payments):
                        payments_data.append({
                            'user_id': user_id,
                            'price': np.random.normal(4000, 1500),
                            'product_category': np.random.choice(product_categories, p=[0.1, 0.3, 0.1, 0.2, 0.2, 0.1]),
                            'timestamp': np.random.choice(dates)
                        })

                elif group_name == 'affluent':
                    n_actions = np.random.randint(10, 30)
                    for i in range(n_actions):
                        market_data.append({
                            'user_id': user_id,
                            'action_type': np.random.choice(['view', 'click', 'clickout'], p=[0.6, 0.2, 0.2]),
                            'subdomain': np.random.choice(['u2i', 'search', 'catalog'], p=[0.2, 0.6, 0.2]),
                            'product_category': np.random.choice(product_categories, p=[0.2, 0.2, 0.3, 0.1, 0.1, 0.1]),
                            'timestamp': np.random.choice(dates)
                        })
                    n_payments = np.random.randint(15, 40)
                    for i in range(n_payments):
                        payments_data.append({
                            'user_id': user_id,
                            'price': np.random.normal(8000, 3000),
                            'product_category': np.random.choice(product_categories, p=[0.2, 0.2, 0.3, 0.1, 0.1, 0.1]),
                            'timestamp': np.random.choice(dates)
                        })

                elif group_name == 'researchers':
                    n_actions = np.random.randint(30, 100)
                    for i in range(n_actions):
                        market_data.append({
                            'user_id': user_id,
                            'action_type': np.random.choice(['view', 'click', 'clickout'], p=[0.3, 0.4, 0.3]),
                            'subdomain': np.random.choice(['u2i', 'search', 'catalog'], p=[0.2, 0.7, 0.1]),
                            'product_category': np.random.choice(product_categories, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1]),
                            'timestamp': np.random.choice(dates)
                        })
                    n_payments = np.random.randint(3, 10)
                    for i in range(n_payments):
                        payments_data.append({
                            'user_id': user_id,
                            'price': np.random.normal(3000, 1000),
                            'product_category': np.random.choice(product_categories, p=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1]),
                            'timestamp': np.random.choice(dates)
                        })

                elif group_name == 'impulsive':
                    n_actions = np.random.randint(25, 70)
                    for i in range(n_actions):
                        market_data.append({
                            'user_id': user_id,
                            'action_type': np.random.choice(['view', 'click', 'clickout'], p=[0.2, 0.3, 0.5]),
                            'subdomain': np.random.choice(['u2i', 'search', 'catalog'], p=[0.7, 0.2, 0.1]),
                            'product_category': np.random.choice(product_categories, p=[0.4, 0.1, 0.1, 0.2, 0.1, 0.1]),
                            'timestamp': np.random.choice(dates)
                        })
                    n_payments = np.random.randint(10, 25)
                    for i in range(n_payments):
                        payments_data.append({
                            'user_id': user_id,
                            'price': np.random.normal(2500, 1200),
                            'product_category': np.random.choice(product_categories, p=[0.4, 0.1, 0.1, 0.2, 0.1, 0.1]),
                            'timestamp': np.random.choice(dates)
                        })

                elif group_name == 'conservative':
                    n_actions = np.random.randint(5, 20)
                    for i in range(n_actions):
                        market_data.append({
                            'user_id': user_id,
                            'action_type': np.random.choice(['view', 'click', 'clickout'], p=[0.7, 0.2, 0.1]),
                            'subdomain': np.random.choice(['u2i', 'search', 'catalog'], p=[0.2, 0.3, 0.5]),
                            'product_category': np.random.choice(product_categories, p=[0.1, 0.4, 0.1, 0.2, 0.1, 0.1]),
                            'timestamp': np.random.choice(dates)
                        })
                    n_payments = np.random.randint(2, 8)
                    for i in range(n_payments):
                        payments_data.append({
                            'user_id': user_id,
                            'price': np.random.normal(5000, 2000),
                            'product_category': np.random.choice(product_categories, p=[0.1, 0.4, 0.1, 0.2, 0.1, 0.1]),
                            'timestamp': np.random.choice(dates)
                        })

                elif group_name == 'digital_natives':
                    n_actions = np.random.randint(40, 120)
                    for i in range(n_actions):
                        market_data.append({
                            'user_id': user_id,
                            'action_type': np.random.choice(['view', 'click', 'clickout'], p=[0.4, 0.4, 0.2]),
                            'subdomain': np.random.choice(['u2i', 'search', 'catalog'], p=[0.5, 0.3, 0.2]),
                            'product_category': np.random.choice(product_categories, p=[0.3, 0.2, 0.2, 0.1, 0.1, 0.1]),
                            'timestamp': np.random.choice(dates)
                        })
                    n_payments = np.random.randint(8, 20)
                    for i in range(n_payments):
                        payments_data.append({
                            'user_id': user_id,
                            'price': np.random.normal(1500, 600),
                            'product_category': np.random.choice(product_categories, p=[0.3, 0.2, 0.2, 0.1, 0.1, 0.1]),
                            'timestamp': np.random.choice(dates)
                        })

                elif group_name == 'business_owners':
                    n_actions = np.random.randint(20, 60)
                    for i in range(n_actions):
                        market_data.append({
                            'user_id': user_id,
                            'action_type': np.random.choice(['view', 'click', 'clickout'], p=[0.5, 0.3, 0.2]),
                            'subdomain': np.random.choice(['u2i', 'search', 'catalog'], p=[0.3, 0.5, 0.2]),
                            'product_category': np.random.choice(product_categories, p=[0.2, 0.2, 0.3, 0.1, 0.1, 0.1]),
                            'timestamp': np.random.choice(dates)
                        })
                    n_payments = np.random.randint(20, 50)
                    for i in range(n_payments):
                        payments_data.append({
                            'user_id': user_id,
                            'price': np.random.normal(10000, 5000),
                            'product_category': np.random.choice(product_categories, p=[0.2, 0.2, 0.3, 0.1, 0.1, 0.1]),
                            'timestamp': np.random.choice(dates)
                        })

        market_df = pd.DataFrame(market_data)
        payments_df = pd.DataFrame(payments_data)
        retail_df = pd.DataFrame(retail_data)

        print("Созданы реалистичные демо-данные с временными метками и категориями")
        return users_df, market_df, payments_df, retail_df
