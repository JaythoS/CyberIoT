#!/usr/bin/env python3
"""
DDoS Tespit Modeli - OPTİMİZE EDİLMİŞ
Daha fazla veri ile ama optimize edilmiş şekilde
"""

import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
import time
import joblib
from tqdm import tqdm

warnings.filterwarnings('ignore')

class OptimizedDDoSModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.model = None
        self.scaler = None

    def load_optimized_dataset(self, num_files=50):
        """Optimize edilmiş veriseti yükle - daha fazla dosya ama hızlı"""
        print(f"🚀 OPTİMİZE EDİLMİŞ VERİSETİ YÜKLENİYOR...")
        print(f"📁 {num_files} dosya kullanılacak")
        print(f"💾 Tahmini boyut: ~4GB")

        start_time = time.time()

        # Tüm DDoS saldırı tipleri
        ddos_attacks = [
            'DDoS-UDP_Flood', 'DDoS-TCP_Flood', 'DDoS-ICMP_Flood', 'DDoS-SYN_Flood',
            'DDoS-PSHACK_Flood', 'DDoS-RSTFINFlood', 'DDoS-SynonymousIP_Flood',
            'DDoS-ICMP_Fragmentation', 'DDoS-UDP_Fragmentation', 'DDoS-ACK_Fragmentation',
            'DDoS-HTTP_Flood', 'DDoS-SlowLoris'
        ]

        # CSV dosyalarını bul
        csv_files = glob.glob(os.path.join(self.data_path, "part-*.csv"))
        csv_files.sort()

        # İlk num_files dosyayı al
        selected_files = csv_files[:num_files]
        print(f"📂 Seçilen dosyalar: 1-{num_files} / {len(csv_files)}")

        data_chunks = []
        total_samples = 0
        ddos_count = 0
        benign_count = 0

        # Dosyaları yükle
        for i, file in enumerate(tqdm(selected_files, desc="Dosyalar yükleniyor")):
            try:
                # Tüm sütunları oku (daha kapsamlı analiz için)
                df_chunk = pd.read_csv(file, low_memory=False)

                # DDoS ve Benign verilerini filtrele
                mask = (df_chunk['label'].isin(ddos_attacks)) | (df_chunk['label'] == 'BenignTraffic')
                df_filtered = df_chunk[mask].copy()

                # Binary label oluştur
                df_filtered['target'] = df_filtered['label'].apply(
                    lambda x: 1 if x in ddos_attacks else 0
                )

                # Sayıları güncelle
                chunk_ddos = len(df_filtered[df_filtered['target'] == 1])
                chunk_benign = len(df_filtered[df_filtered['target'] == 0])
                ddos_count += chunk_ddos
                benign_count += chunk_benign
                total_samples += len(df_filtered)

                data_chunks.append(df_filtered)

                # Her 10 dosyada bir rapor ver
                if (i + 1) % 10 == 0:
                    print(f"📊 {i+1}/{len(selected_files)} dosya yüklendi")
                    print(f"   Toplam: {total_samples:,} | DDoS: {ddos_count:,} | Benign: {benign_count:,}")

            except Exception as e:
                print(f"❌ Hata ({file}): {e}")
                continue

        if not data_chunks:
            raise ValueError("Hiç veri yüklenemedi!")

        # Tüm verileri birleştir
        print("\n🔗 Veriler birleştiriliyor...")
        self.df = pd.concat(data_chunks, ignore_index=True)

        end_time = time.time()
        loading_time = end_time - start_time

        print(f"\n✅ OPTİMİZE EDİLMİŞ VERİSETİ YÜKLENDI!")
        print(f"⏱️  Yükleme süresi: {loading_time/60:.1f} dakika")
        print(f"📊 Toplam örnek: {len(self.df):,}")
        print(f"🔥 DDoS saldırıları: {ddos_count:,}")
        print(f"🛡️  Benign trafik: {benign_count:,}")
        print(f"📈 DDoS oranı: {ddos_count/len(self.df)*100:.2f}%")

        # DDoS türleri dağılımı
        print(f"\n🎯 DDOS SALDIRI TÜRLERİ (ilk 10):")
        ddos_types = self.df[self.df['target'] == 1]['label'].value_counts()
        for attack_type, count in ddos_types.head(10).items():
            print(f"   {attack_type:<25}: {count:,}")

        return self.df

    def prepare_optimized_data(self):
        """Optimize edilmiş veri hazırlama"""
        print(f"\n🔧 OPTİMİZE EDİLMİŞ VERİ HAZıRLAMA")

        if self.df is None:
            raise ValueError("Veri yüklenmedi!")

        # Tüm sayısal özellikleri al
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()

        # Label ve target'ı çıkar
        features = [col for col in numeric_columns if col not in ['target', 'binary_label', 'Unnamed: 0']]

        print(f"📊 Toplam özellik: {len(features)}")

        # Özellikler ve hedef
        X = self.df[features].copy()
        y = self.df['target'].copy()

        print(f"📊 Özellik matrisi: {X.shape}")

        # Sonsuz ve NaN değerleri temizle
        print("🧹 Veri temizleniyor...")
        X = X.replace([np.inf, -np.inf], np.nan)

        # NaN değerleri sütun ortalaması ile doldur
        nan_columns = []
        for col in X.columns:
            if X[col].isnull().any():
                mean_val = X[col].mean()
                if not pd.isna(mean_val):  # Sadece geçerli ortalamalarla doldur
                    X[col] = X[col].fillna(mean_val)
                    nan_columns.append(col)

        if nan_columns:
            print(f"   {len(nan_columns)} sütun NaN ile dolduruldu")

        # Kalan NaN değerleri 0 ile doldur
        X = X.fillna(0)

        print(f"✅ Veri temizlendi: {X.isnull().sum().sum()} NaN kaldı")

        # %75 eğitim, %25 test bölmesi
        print("📦 Veri bölünüyor (75-25)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        print(f"✅ Veri hazırlandı:")
        print(f"   Eğitim seti: {X_train.shape[0]:,} örnek")
        print(f"   Test seti: {X_test.shape[0]:,} örnek")
        print(f"   Özellik sayısı: {X_train.shape[1]}")

        # Sınıf dağılımı
        train_ddos = sum(y_train == 1)
        train_benign = sum(y_train == 0)
        test_ddos = sum(y_test == 1)
        test_benign = sum(y_test == 0)

        print(f"\n📈 Sınıf Dağılımı:")
        print(f"   Eğitim - DDoS: {train_ddos:,} ({train_ddos/len(y_train)*100:.2f}%)")
        print(f"   Eğitim - Benign: {train_benign:,} ({train_benign/len(y_train)*100:.2f}%)")
        print(f"   Test - DDoS: {test_ddos:,} ({test_ddos/len(y_test)*100:.2f}%)")
        print(f"   Test - Benign: {test_benign:,} ({test_benign/len(y_test)*100:.2f}%)")

        # Ölçeklendirme
        print("⚖️  Özellikler ölçeklendiriliyor...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, features

    def train_optimized_model(self, X_train, y_train):
        """Optimize edilmiş Random Forest eğitimi"""
        print(f"\n🌲 OPTİMİZE EDİLMİŞ RANDOM FOREST EĞİTİMİ")

        # Daha fazla özellik için optimize edilmiş parametreler
        self.model = RandomForestClassifier(
            n_estimators=150,          # Orta ağaç sayısı
            max_depth=25,              # Orta derinlik
            min_samples_split=15,      # Overfitting önleme
            min_samples_leaf=8,        # Daha kararlı yapraklar
            max_features='sqrt',       # Özellik optimizasyonu
            bootstrap=True,            # Bootstrap
            oob_score=True,            # Out-of-bag skor
            class_weight='balanced',   # Sınıf dengesizliği
            random_state=42,
            n_jobs=-1                  # Paralel işlem
        )

        print("🚀 Eğitim başlıyor...")
        print(f"📊 Eğitim verisi: {X_train.shape}")
        print(f"🌳 Ağaç sayısı: {self.model.n_estimators}")
        print(f"⚙️  Maksimum derinlik: {self.model.max_depth}")

        start_time = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time

        print(f"✅ Eğitim tamamlandı! ({training_time/60:.1f} dakika)")
        print(f"📊 Out-of-Bag skoru: {self.model.oob_score_:.4f}")

        return self.model

    def evaluate_optimized_model(self, X_test, y_test, features):
        """Optimize edilmiş model değerlendirmesi"""
        print(f"\n📊 OPTİMİZE EDİLMİŞ MODEL DEĞERLENDİRMESİ")

        # Tahminler
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        # Temel metrikler
        accuracy = accuracy_score(y_test, y_pred)

        print(f"🎯 Test Doğruluğu: {accuracy:.4f} ({accuracy*100:.2f}%)")

        # Detaylı sınıflandırma raporu
        print(f"\n📋 Detaylı Sınıflandırma Raporu:")
        print(classification_report(y_test, y_pred,
                                   target_names=['Benign', 'DDoS'],
                                   digits=4))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        print(f"\n🔢 Confusion Matrix:")
        print(f"   True Negatives (Benign doğru):    {tn:,}")
        print(f"   False Positives (Benign yanlış):  {fp:,}")
        print(f"   False Negatives (DDoS kaçırılan): {fn:,}")
        print(f"   True Positives (DDoS doğru):      {tp:,}")

        # DDoS için özel metrikler
        detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        print(f"\n🎯 DDOS TESPİT METRİKLERİ:")
        print(f"   Tespit Oranı (Recall):    {detection_rate:.4f} ({detection_rate*100:.2f}%)")
        print(f"   Yanlış Alarm Oranı:       {false_positive_rate:.4f} ({false_positive_rate*100:.2f}%)")
        print(f"   Kesinlik (Precision):     {precision:.4f} ({precision*100:.2f}%)")

        # Özellik önemleri
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)

        print(f"\n🎯 En Önemli 15 Özellik:")
        for i, (idx, row) in enumerate(feature_importance.head(15).iterrows(), 1):
            print(f"   {i:2d}. {row['Feature']:<25} ({row['Importance']:.4f})")

        # CSV sonuçları oluştur
        results_df = pd.DataFrame({
            'Feature': features,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)

        results_df.to_csv('feature_importance_optimized.csv', index=False)
        print(f"📄 Özellik önemleri kaydedildi: feature_importance_optimized.csv")

        return {
            'accuracy': accuracy,
            'detection_rate': detection_rate,
            'false_positive_rate': false_positive_rate,
            'precision': precision,
            'confusion_matrix': cm,
            'feature_importance': feature_importance
        }

    def save_optimized_model(self, features, filename='ddos_optimized_model.pkl'):
        """Optimize edilmiş modeli kaydet"""
        print(f"\n💾 OPTİMİZE EDİLMİŞ MODEL KAYDEDİLİYOR: {filename}")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'features': features,
            'dataset_info': {
                'total_samples': len(self.df),
                'feature_count': len(features),
                'model_type': 'Optimized Random Forest'
            }
        }

        joblib.dump(model_data, filename)
        print(f"✅ Model başarıyla kaydedildi!")
        return filename

def main():
    """Ana fonksiyon"""
    print("🚀 DDOS TESPİT MODELİ - OPTİMİZE EDİLMİŞ")
    print("Daha fazla veri ile optimize edilmiş şekilde")
    print("="*80)

    DATA_PATH = '/Users/enes/Desktop/sibers/data/external/wataiData/csv/CICIoT2023'

    try:
        # Model oluştur
        optimized_model = OptimizedDDoSModel(DATA_PATH)

        # Optimize edilmiş verisetini yükle (50 dosya)
        df = optimized_model.load_optimized_dataset(num_files=50)

        # Veriyi hazırla
        X_train, X_test, y_train, y_test, features = optimized_model.prepare_optimized_data()

        # Model eğitimi
        model = optimized_model.train_optimized_model(X_train, y_train)

        # Model değerlendirme
        results = optimized_model.evaluate_optimized_model(X_test, y_test, features)

        # Modeli kaydet
        filename = optimized_model.save_optimized_model(features)

        print(f"\n🎉 OPTİMİZE EDİLMİŞ MODEL BAŞARIYLA OLUŞTURULDU!")
        print(f"📄 Model dosyası: {filename}")
        print(f"📊 Sonuçlar:")
        print(f"   Doğruluk: {results['accuracy']*100:.2f}%")
        print(f"   DDoS Tespit Oranı: {results['detection_rate']*100:.2f}%")
        print(f"   Yanlış Alarm Oranı: {results['false_positive_rate']*100:.2f}%")
        print(f"   Özellik Sayısı: {len(features)}")

    except Exception as e:
        print(f"❌ Hata: {e}")
        raise

if __name__ == "__main__":
    main()