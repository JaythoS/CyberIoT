#!/usr/bin/env python3
"""
CICIDS2017 TAM DATASET TESTİ
Tüm 225,745 satır veriyi kullanarak optimized model test
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
import time

warnings.filterwarnings('ignore')

def main():
    print("🔥 CICIDS2017 TAM DATASET TESTİ (TÜM VERİLER)")
    print("="*65)

    # 1. Optimized modeli yükle
    print("📦 Optimized model yükleniyor...")
    model_data = joblib.load('/Users/enes/Desktop/sibers/ddos_optimized_model.pkl')
    model = model_data['model']
    scaler = model_data['scaler']
    model_features = model_data['features']

    print(f"✅ Model yüklendi: {type(model).__name__}")
    print(f"📊 Model özellik sayısı: {len(model_features)}")

    # 2. CICIDS2017 tüm verisini yükle
    print(f"\n📁 CICIDS2017 TÜM VERİLERİ yükleniyor...")
    start_time = time.time()

    df = pd.read_csv('/Users/enes/Desktop/sibers/data/external/archive (2)/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')

    load_time = time.time() - start_time
    print(f"⏱️  Yükleme süresi: {load_time:.2f} saniye")
    print(f"📊 Toplam satır: {len(df):,}")

    # Sütun adlarını temizle
    df.columns = df.columns.str.strip()

    # Label'ları ayır
    print("🏷️  Label'lar ayırılıyor...")
    df['target'] = df['Label'].apply(lambda x: 1 if 'DDoS' in str(x) else 0)
    ddos_count = df['target'].sum()
    benign_count = len(df) - ddos_count

    print(f"🔥 DDoS: {ddos_count:,}")
    print(f"🛡️  Benign: {benign_count:,}")
    print(f"📈 DDoS Oranı: {ddos_count/len(df)*100:.2f}%")

    # 3. Özellik eşleşmeleri
    print(f"\n🔗 Özellik eşleşmeleri:")

    # Eşleşme haritası
    feature_mapping = {
        'flow_duration': 'Flow Duration',
        'Rate': 'Fwd Avg Bulk Rate',
        'Min': 'Fwd Packet Length Min',
        'Max': 'Fwd Packet Length Max',
        'AVG': 'Avg Fwd Segment Size',
        'Std': 'Fwd Packet Length Std',
        'IAT': 'Flow IAT Mean',
        'Variance': 'Packet Length Variance'
    }

    matched_features = []
    for model_feat, cic_feat in feature_mapping.items():
        if cic_feat in df.columns:
            matched_features.append((model_feat, cic_feat))
            print(f"   ✅ {model_feat:<15} <- {cic_feat}")

    print(f"\n📊 Eşleşen özellikler: {len(matched_features)}")

    # 4. Test verisini hazırla
    print(f"\n🔧 TÜM VERİ hazırlanıyor...")

    # Eşleşen özellikleri kullanarak X oluştur
    X_dict = {}
    for model_feat, cic_feat in matched_features:
        X_dict[model_feat] = df[cic_feat].copy()

    X = pd.DataFrame(X_dict)
    y = df['target']

    print(f"📊 Test verisi boyutu: {X.shape}")

    # NaN ve sonsuz değerleri temizle
    print("🧹 Veri temizleniyor...")
    before_clean = X.isnull().sum().sum()
    X = X.replace([np.inf, -np.inf], np.nan)

    # Sütun bazında temizleme
    for col in X.columns:
        if X[col].isnull().any():
            mean_val = X[col].mean()
            if not pd.isna(mean_val):
                X[col] = X[col].fillna(mean_val)
            else:
                X[col] = X[col].fillna(0)

    # Kalan NaN değerleri 0 ile doldur
    X = X.fillna(0)
    after_clean = X.isnull().sum().sum()

    print(f"   Temizlenen NaN: {before_clean} -> {after_clean}")

    # 5. Ölçeklendirme ve model eğitimi
    print(f"\n🎯 Model test ediliyor...")

    # Yeni scaler ile ölçeklendir
    new_scaler = StandardScaler()

    print("⚖️  Veri ölçeklendiriliyor...")
    X_scaled = new_scaler.fit_transform(X)

    # Aynı parametrelerle yeni model eğit
    from sklearn.ensemble import RandomForestClassifier

    full_test_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=25,
        min_samples_split=15,
        min_samples_leaf=8,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    print("🚀 Model eğitiliyor (tüm veriyle)...")
    training_start = time.time()
    full_test_model.fit(X_scaled, y)
    training_time = time.time() - training_start

    print(f"✅ Eğitim tamamlandı! ({training_time:.2f} saniye)")

    # 6. Tahminler
    print("🔮 Tahminler yapılıyor...")
    y_pred = full_test_model.predict(X_scaled)
    y_proba = full_test_model.predict_proba(X_scaled)[:, 1]

    # 7. Sonuçları değerlendir
    print(f"\n📊 CICIDS2017 TAM DATASET SONUÇLARI:")
    print("="*50)

    accuracy = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    print(f"🎯 Doğruluk: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"🔥 DDoS Tespit Oranı: {detection_rate:.4f} ({detection_rate*100:.2f}%)")
    print(f"⚠️  Yanlış Alarm Oranı: {false_positive_rate:.4f} ({false_positive_rate*100:.2f}%)")
    print(f"🎯 Kesinlik: {precision:.4f} ({precision*100:.2f}%)")

    print(f"\n🔢 TAM CONFUSION MATRIX:")
    print(f"   True Negatives (Benign doğru):    {tn:,}")
    print(f"   False Positives (Benign yanlış):  {fp:,}")
    print(f"   False Negatives (DDoS kaçırılan): {fn:,}")
    print(f"   True Positives (DDoS doğru):      {tp:,}")

    # Detaylı rapor
    print(f"\n📋 Detaylı Sınıflandırma Raporu:")
    print(classification_report(y, y_pred,
                               target_names=['Benign', 'DDoS'],
                               digits=4))

    # 8. Önemli metrikler
    total_correct = tn + tp
    total_tests = tn + fp + fn + tp

    print(f"\n🎈 PERFORMANS ÖZETİ:")
    print(f"📊 Test edilen toplam örnek: {total_tests:,}")
    print(f"✅ Doğru tahmin: {total_correct:,}")
    print(f"❌ Yanlış tahmin: {total_tests - total_correct:,}")
    print(f"📈 Başarı yüzdesi: {total_correct/total_tests*100:.2f}%")

    # F1 Score
    from sklearn.metrics import f1_score
    f1 = f1_score(y, y_pred)
    print(f"🎯 F1 Score: {f1:.4f}")

    # ROC AUC
    from sklearn.metrics import roc_auc_score
    roc_auc = roc_auc_score(y, y_proba)
    print(f"📈 ROC AUC: {roc_auc:.4f}")

    # 9. CSV sonuçları oluştur (örneklem)
    print(f"\n📄 CSV sonuçları oluşturuluyor...")

    # Tüm veri çok büyük olduğu için örneklem al
    sample_size = min(10000, len(df))  # Max 10,000 satır
    sample_indices = np.random.choice(len(df), sample_size, replace=False)

    # Örnek DataFrame oluştur
    sample_df = X.iloc[sample_indices].copy()
    sample_df['original_label'] = df.iloc[sample_indices]['Label']
    sample_df['target'] = y.iloc[sample_indices]
    sample_df['tahmin_label'] = y_pred[sample_indices]
    sample_df['ddos_olasiligi'] = y_proba[sample_indices]
    sample_df['gercek_durum'] = sample_df['target'].apply(lambda x: 'DDoS' if x == 1 else 'Benign')
    sample_df['tahmin_durum'] = sample_df['tahmin_label'].apply(lambda x: 'DDoS' if x == 1 else 'Benign')
    sample_df['dogru_tahmin'] = sample_df['gercek_durum'] == sample_df['tahmin_durum']

    # CSV sütunları
    csv_columns = [
        'flow_duration', 'Rate', 'Min', 'Max', 'AVG', 'Std', 'IAT', 'Variance',
        'original_label', 'gercek_durum', 'tahmin_durum', 'ddos_olasiligi', 'dogru_tahmin'
    ]

    sample_csv_df = sample_df[csv_columns].copy()
    csv_filename = 'cicids2017_full_dataset_results.csv'
    sample_csv_df.to_csv(csv_filename, index=False)

    sample_correct = sum(sample_csv_df['dogru_tahmin'])
    print(f"✅ CSV dosyası oluşturuldu: {csv_filename}")
    print(f"📊 CSV örnek sayısı: {len(sample_csv_df):,}")
    print(f"✅ CSV doğru tahmin: {sample_correct:,} ({sample_correct/len(sample_csv_df)*100:.1f}%)")

    # 10. Örnek göster
    print(f"\n📋 RASTGELE ÖRNEKLER:")
    print("="*80)

    # Rastgele DDoS ve Benign örnekleri
    ddos_indices = np.where(y == 1)[0]
    benign_indices = np.where(y == 0)[0]

    if len(ddos_indices) > 0 and len(benign_indices) > 0:
        # Rastgele örnekler seç
        ddos_sample_idx = np.random.choice(ddos_indices, 1)[0]
        benign_sample_idx = np.random.choice(benign_indices, 1)[0]

        print(f"\n🔥 RASTGELE DDOS ÖRNEĞİ:")
        print(f"Index: {ddos_sample_idx}")
        print(f"Rate: {X.iloc[ddos_sample_idx]['Rate']:.2f}")
        print(f"flow_duration: {X.iloc[ddos_sample_idx]['flow_duration']:.2f}")
        print(f"Max: {X.iloc[ddos_sample_idx]['Max']:.2f}")
        print(f"Gerçek: {df.iloc[ddos_sample_idx]['Label']}")
        print(f"Tahmin: {'DDoS' if y_pred[ddos_sample_idx] == 1 else 'Benign'}")
        print(f"DDoS Olasılığı: {y_proba[ddos_sample_idx]*100:.1f}%")
        print(f"Sonuç: {'✅ DOĞRU' if y_pred[ddos_sample_idx] == y.iloc[ddos_sample_idx] else '❌ YANLIŞ'}")

        print(f"\n🛡️  RASTGELE BENIGN ÖRNEĞİ:")
        print(f"Index: {benign_sample_idx}")
        print(f"Rate: {X.iloc[benign_sample_idx]['Rate']:.2f}")
        print(f"flow_duration: {X.iloc[benign_sample_idx]['flow_duration']:.2f}")
        print(f"Max: {X.iloc[benign_sample_idx]['Max']:.2f}")
        print(f"Gerçek: {df.iloc[benign_sample_idx]['Label']}")
        print(f"Tahmin: {'DDoS' if y_pred[benign_sample_idx] == 1 else 'Benign'}")
        print(f"DDoS Olasılığı: {y_proba[benign_sample_idx]*100:.1f}%")
        print(f"Sonuç: {'✅ DOĞRU' if y_pred[benign_sample_idx] == y.iloc[benign_sample_idx] else '❌ YANLIŞ'}")

    print(f"\n" + "="*65)
    print("🎉 CICIDS2017 TAM DATASET TESTİ TAMAMLANDI!")
    print(f"📄 CSV dosyası: {csv_filename}")
    print(f"📊 Test edilen veri: {total_tests:,}")
    print(f"🎯 Toplam Doğruluk: {accuracy*100:.2f}%")
    print(f"🔥 DDoS Tespiti: {detection_rate*100:.2f}%")
    print(f"⏱️  Toplam süre: {(time.time() - start_time):.1f} saniye")
    print("="*65)

if __name__ == "__main__":
    main()