#!/usr/bin/env python3
"""
DDoS Model Veri Analizi ve CSV Çıktı
Modelde kullanılan veriler ve test sonuçları
"""

import pandas as pd
import numpy as np
import os
import glob
import joblib
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

class DDoSDataAnalysis:
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = None
        self.scaler = None
        self.features = None

    def show_model_features(self):
        """Modelde kullanılan özellikleri göster"""
        print("🔍 MODELDE KULLANILAN VERİLER")
        print("="*60)

        try:
            model_data = joblib.load('ddos_fast_model.pkl')
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.features = model_data['features']
        except:
            print("❌ Model bulunamadı!")
            return False

        print(f"✅ Model bilgileri yüklendi")
        print(f"Model tipi: {type(self.model).__name__}")
        print(f"Özellik sayısı: {len(self.features)}")

        print(f"\n📊 MODELDE KULLANILAN 20 KRİTİK ÖZELLİK:")
        print("-" * 60)

        for i, feature in enumerate(self.features, 1):
            description = self.get_feature_description(feature)
            print(f"{i:2d}. {feature:<20} | {description}")

        print("-" * 60)
        return True

    def get_feature_description(self, feature):
        """Özellik açıklamaları"""
        descriptions = {
            'Rate': 'Trafik hızı (packets/saniye)',
            'Srate': 'Kaynak tarafı trafik hızı',
            'Drate': 'Hedef tarafı trafik hızı',
            'flow_duration': 'Akış süresi (saniye)',
            'Header_Length': 'Paket başlık uzunluğu',
            'Tot size': 'Toplam veri boyutu',
            'Number': 'Paket sayısı',
            'syn_flag_number': 'SYN bayrak sayısı',
            'ack_flag_number': 'ACK bayrak sayısı',
            'rst_flag_number': 'RST bayrak sayısı',
            'fin_flag_number': 'FIN bayrak sayısı',
            'psh_flag_number': 'PSH bayrak sayısı',
            'Max': 'Maksimum değer',
            'Min': 'Minimum değer',
            'AVG': 'Ortalama değer',
            'Std': 'Standart sapma',
            'Variance': 'Varyans',
            'TCP': 'TCP protokolü (0/1)',
            'UDP': 'UDP protokolü (0/1)',
            'ICMP': 'ICMP protokolü (0/1)'
        }
        return descriptions.get(feature, 'Ağ trafiği özelliği')

    def load_test_data(self, num_samples=200):
        """Test verileri yükle"""
        print(f"\n📥 TEST VERİLERİ YÜKLENİYOR (Toplam {num_samples} örnek)...")

        # DDoS saldırı tipleri
        ddos_attacks = [
            'DDoS-UDP_Flood', 'DDoS-TCP_Flood', 'DDoS-ICMP_Flood', 'DDoS-SYN_Flood',
            'DDoS-PSHACK_Flood', 'DDoS-RSTFINFlood', 'DDoS-SynonymousIP_Flood',
            'DDoS-ICMP_Fragmentation', 'DDoS-UDP_Fragmentation', 'DDoS-ACK_Fragmentation'
        ]

        csv_files = glob.glob(os.path.join(self.data_path, "part-*.csv"))

        if len(csv_files) == 0:
            print("❌ Veri dosyaları bulunamadı!")
            return None

        # İlk 5 dosyadan veri al
        test_files = csv_files[:5]
        all_samples = []

        for file in test_files:
            try:
                df = pd.read_csv(file, low_memory=False)

                # DDoS ve Benign verilerini filtrele
                mask = (df['label'].isin(ddos_attacks)) | (df['label'] == 'BenignTraffic')
                df_filtered = df[mask].copy()

                # Binary label oluştur
                df_filtered['gercek_label'] = df_filtered['label'].apply(
                    lambda x: 1 if x in ddos_attacks else 0
                )

                all_samples.append(df_filtered)

                if len(pd.concat(all_samples, ignore_index=True)) > num_samples:
                    break

            except Exception as e:
                print(f"Dosya okuma hatası: {e}")
                continue

        if not all_samples:
            print("❌ Hiç örnek yüklenemedi!")
            return None

        df_all = pd.concat(all_samples, ignore_index=True)

        # Dengeli örnek seç
        max_per_class = num_samples // 2
        ddos_samples = df_all[df_all['gercek_label'] == 1].head(max_per_class)
        benign_samples = df_all[df_all['gercek_label'] == 0].head(max_per_class)

        df_test = pd.concat([ddos_samples, benign_samples], ignore_index=True)

        print(f"✅ {len(df_test)} örnek yüklendi")
        print(f"   DDoS örnekleri: {len(ddos_samples)}")
        print(f"   Benign örnekleri: {len(benign_samples)}")

        return df_test

    def create_csv_results(self, df_test):
        """CSV formatında test sonuçları oluştur"""
        print(f"\n📊 MODEL TESTİ BAŞLATILIYOR...")

        # Özellikleri hazırla
        X_test = df_test[self.features].copy()

        # Sonsuz ve NaN değerleri temizle
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.fillna(X_test.mean())

        # Ölçeklendir
        X_test_scaled = self.scaler.transform(X_test)

        # Tahminler
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        # Sonuç DataFrame'i oluştur
        results_df = df_test.copy()
        results_df['tahmin_label'] = y_pred
        results_df['ddos_olasiligi'] = y_proba
        results_df['gercek_durum'] = results_df['gercek_label'].apply(
            lambda x: 'DDoS' if x == 1 else 'Benign'
        )
        results_df['tahmin_durum'] = results_df['tahmin_label'].apply(
            lambda x: 'DDoS' if x == 1 else 'Benign'
        )
        results_df['dogru_tahmin'] = results_df['gercek_durum'] == results_df['tahmin_durum']

        # CSV için sütunları seç
        csv_columns = [
            # Özellikler
            'Rate', 'Srate', 'Drate', 'flow_duration', 'Header_Length',
            'Tot size', 'Number', 'syn_flag_number', 'ack_flag_number',
            'rst_flag_number', 'fin_flag_number', 'psh_flag_number',
            'Max', 'Min', 'AVG', 'Std', 'Variance', 'TCP', 'UDP', 'ICMP',

            # Gerçek veriler
            'label', 'gercek_durum',

            # Tahminler
            'tahmin_durum', 'ddos_olasiligi', 'dogru_tahmin'
        ]

        # Sadece mevcut sütunları al
        available_columns = [col for col in csv_columns if col in results_df.columns]
        final_csv_df = results_df[available_columns].copy()

        # CSV'ye yaz
        csv_filename = 'ddos_test_results.csv'
        final_csv_df.to_csv(csv_filename, index=False)

        print(f"✅ CSV dosyası oluşturuldu: {csv_filename}")
        print(f"   Toplam kayıt: {len(final_csv_df)}")
        print(f"   Sütun sayısı: {len(final_csv_df.columns)}")

        return final_csv_df, csv_filename

    def show_csv_preview(self, df_csv, filename):
        """CSV dosyasının önizlemesini göster"""
        print(f"\n📋 CSV DOSYASI ÖNİZLEMESİ: {filename}")
        print("="*100)

        # İlk 10 kaydı göster
        print(f"\n📝 İLK 10 KAYIT:")
        print("-" * 100)

        # Sütunları daha iyi göstermek için format
        pd.set_on('display.max_columns', None)
        pd.set_option('display.width', 100)
        pd.set_option('display.float_format', '{:.2f}'.format)

        preview_df = df_csv.head(10).copy()

        # Önemli sütunları vurgula
        important_cols = ['Rate', 'Header_Length', 'Tot size', 'gercek_durum', 'tahmin_durum', 'ddos_olasiligi', 'dogru_tahmin']

        for col in important_cols:
            if col in preview_df.columns:
                if col == 'ddos_olasiligi':
                    preview_df[col] = preview_df[col].apply(lambda x: f"{x*100:.1f}%")
                elif col == 'dogru_tahmin':
                    preview_df[col] = preview_df[col].apply(lambda x: "✅ DOĞRU" if x else "❌ YANLIŞ")

        print(preview_df.to_string(index=False))

        # Sütun bilgileri
        print(f"\n📊 CSV SÜTUN BİLGİLERİ:")
        print(f"Toplam sütun: {len(df_csv.columns)}")

        feature_cols = [col for col in df_csv.columns if col in self.features]
        result_cols = [col for col in df_csv.columns if col not in self.features]

        print(f"Özellik sütunları: {len(feature_cols)}")
        print(f"Sonuç sütunları: {len(result_cols)}")

        print(f"\n🎯 SONUÇ ÖZETİ:")
        dogru = sum(df_csv['dogru_tahmin']) if 'dogru_tahmin' in df_csv.columns else 0
        toplam = len(df_csv)
        dogruluk = (dogru / toplam) * 100 if toplam > 0 else 0

        print(f"Doğru tahmin: {dogru}/{toplam} ({dogruluk:.2f}%)")

        if 'gercek_durum' in df_csv.columns and 'tahmin_durum' in df_csv.columns:
            ddos_dogru = len(df_csv[(df_csv['gercek_durum'] == 'DDoS') & (df_csv['tahmin_durum'] == 'DDoS')])
            ddos_toplam = len(df_csv[df_csv['gercek_durum'] == 'DDoS'])
            ddos_oran = (ddos_dogru / ddos_toplam) * 100 if ddos_toplam > 0 else 0
            print(f"DDoS tespit oranı: {ddos_dogru}/{ddos_toplam} ({ddos_oran:.2f}%)")

    def show_sample_analysis(self, df_csv):
        """Örnek analiz göster"""
        print(f"\n🔍 ÖRNEK TRAFİK ANALİZİ:")
        print("="*80)

        # Bir DDoS örneği
        ddos_sample = df_csv[df_csv['gercek_durum'] == 'DDoS'].iloc[0] if len(df_csv[df_csv['gercek_durum'] == 'DDoS']) > 0 else None
        # Bir Benign örneği
        benign_sample = df_csv[df_csv['gercek_durum'] == 'Benign'].iloc[0] if len(df_csv[df_csv['gercek_durum'] == 'Benign']) > 0 else None

        if ddos_sample is not None:
            print(f"\n🔥 DDOS ÖRNEĞİ:")
            print(f"Rate: {ddos_sample.get('Rate', 'N/A')}")
            print(f"Header_Length: {ddos_sample.get('Header_Length', 'N/A')}")
            print(f"Tot size: {ddos_sample.get('Tot size', 'N/A')}")
            print(f"Gerçek: {ddos_sample.get('gercek_durum', 'N/A')}")
            print(f"Tahmin: {ddos_sample.get('tahmin_durum', 'N/A')}")
            print(f"DDoS Olasılığı: {ddos_sample.get('ddos_olasiligi', 0)*100:.1f}%")
            print(f"Sonuç: {'✅ DOĞRU' if ddos_sample.get('dogru_tahmin', False) else '❌ YANLIŞ'}")

        if benign_sample is not None:
            print(f"\n🛡️ BENIGN ÖRNEĞİ:")
            print(f"Rate: {benign_sample.get('Rate', 'N/A')}")
            print(f"Header_Length: {benign_sample.get('Header_Length', 'N/A')}")
            print(f"Tot size: {benign_sample.get('Tot size', 'N/A')}")
            print(f"Gerçek: {benign_sample.get('gercek_durum', 'N/A')}")
            print(f"Tahmin: {benign_sample.get('tahmin_durum', 'N/A')}")
            print(f"DDoS Olasılığı: {benign_sample.get('ddos_olasiligi', 0)*100:.1f}%")
            print(f"Sonuç: {'✅ DOĞRU' if benign_sample.get('dogru_tahmin', False) else '❌ YANLIŞ'}")

def main():
    """Ana fonksiyon"""
    print("🔍 DDOS MODEL VERİ ANALİZİ VE CSV ÇIKTI")
    print("="*80)

    DATA_PATH = '/Users/enes/Desktop/sibers/data/external/wataiData/csv/CICIoT2023'

    analyzer = DDoSDataAnalysis(DATA_PATH)

    # Model özelliklerini göster
    if not analyzer.show_model_features():
        return

    # Test verilerini yükle
    df_test = analyzer.load_test_data(200)
    if df_test is None:
        return

    # CSV sonuçları oluştur
    df_csv, filename = analyzer.create_csv_results(df_test)

    # CSV önizlemesi göster
    analyzer.show_csv_preview(df_csv, filename)

    # Örnek analiz
    analyzer.show_sample_analysis(df_csv)

    print(f"\n" + "="*80)
    print("✅ ANALİZ TAMAMLANDI!")
    print(f"📄 CSV dosyası: {filename}")
    print(f"📊 Toplam kayıt: {len(df_csv)}")
    print("="*80)

if __name__ == "__main__":
    main()