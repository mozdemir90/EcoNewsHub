import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class ModelComparator:
    def __init__(self):
        self.results = {}
        self.test_data = None
        
    def load_test_data(self):
        """Test verilerini yükle"""
        print("📥 Test verileri yükleniyor...")
        
        # Tüm tahminli dosyaları yükle
        files = {
            'TF-IDF': 'data/analiz_sonuclari2_tahminli_TF-IDF.xlsx',
            'Word2Vec': 'data/analiz_sonuclari2_tahminli_w2v.xlsx',
            'Deep Learning': 'data/analiz_sonuclari2_tahminli_DL.xlsx'
        }
        
        self.test_data = {}
        for name, file_path in files.items():
            if os.path.exists(file_path):
                self.test_data[name] = pd.read_excel(file_path)
                print(f"✅ {name}: {len(self.test_data[name])} satır")
            else:
                print(f"❌ {name}: Dosya bulunamadı - {file_path}")
        
        # Orijinal test verisi
        if os.path.exists('data/analiz_sonuclari2.xlsx'):
            self.original_test = pd.read_excel('data/analiz_sonuclari2.xlsx')
            print(f"✅ Orijinal test: {len(self.original_test)} satır")
        else:
            print("❌ Orijinal test verisi bulunamadı")
            return False
            
        return True
    
    def calculate_metrics(self, y_true, y_pred):
        """Metrikleri hesapla"""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'MSE': mse,
            'MAE': mae,
            'R²': r2,
            'RMSE': np.sqrt(mse)
        }
    
    def evaluate_models(self):
        """Tüm modelleri değerlendir"""
        print("\n📊 Model değerlendirmesi başlıyor...")
        
        varliklar = ['dolar_skor', 'altin_skor', 'borsa_skor', 'bitcoin_skor']
        
        for model_name, test_df in self.test_data.items():
            print(f"\n🎯 {model_name} değerlendiriliyor...")
            
            model_results = {}
            
            for varlik in varliklar:
                # Gerçek değerler
                y_true = self.original_test[varlik].values
                
                # Tahmin edilen değerler (farklı sütun isimleri için)
                pred_columns = [col for col in test_df.columns if varlik in col and 'skor' in col and col != varlik]
                
                if pred_columns:
                    # İlk tahmin sütununu kullan
                    y_pred = test_df[pred_columns[0]].values
                    
                    # Metrikleri hesapla
                    metrics = self.calculate_metrics(y_true, y_pred)
                    
                    print(f"  {varlik}: MSE={metrics['MSE']:.4f}, MAE={metrics['MAE']:.4f}, R²={metrics['R²']:.4f}")
                    
                    model_results[varlik] = metrics
                else:
                    print(f"  {varlik}: Tahmin sütunu bulunamadı")
            
            self.results[model_name] = model_results
    
    def create_comparison_table(self):
        """Karşılaştırma tablosu oluştur"""
        print("\n📋 Karşılaştırma tablosu oluşturuluyor...")
        
        comparison_data = []
        
        for model_name, model_results in self.results.items():
            for varlik, metrics in model_results.items():
                comparison_data.append({
                    'Model': model_name,
                    'Varlık': varlik,
                    'MSE': metrics['MSE'],
                    'MAE': metrics['MAE'],
                    'R²': metrics['R²'],
                    'RMSE': metrics['RMSE']
                })
        
        self.comparison_df = pd.DataFrame(comparison_data)
        
        # Excel'e kaydet
        self.comparison_df.to_excel('data/model_comparison.xlsx', index=False)
        print("✅ Karşılaştırma tablosu kaydedildi: data/model_comparison.xlsx")
        
        return self.comparison_df
    
    def create_visualizations(self):
        """Görselleştirmeler oluştur"""
        print("\n📈 Görselleştirmeler oluşturuluyor...")
        
        # Matplotlib ayarları
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performans Karşılaştırması', fontsize=16, fontweight='bold')
        
        # Metrikler
        metrics = ['MSE', 'MAE', 'R²', 'RMSE']
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            
            # Pivot table oluştur
            pivot_data = self.comparison_df.pivot(index='Varlık', columns='Model', values=metric)
            
            # Heatmap
            sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='RdYlBu_r', ax=ax)
            ax.set_title(f'{metric} Karşılaştırması')
            ax.set_xlabel('Model')
            ax.set_ylabel('Varlık')
        
        plt.tight_layout()
        plt.savefig('data/model_comparison_heatmap.png', dpi=300, bbox_inches='tight')
        print("✅ Heatmap kaydedildi: data/model_comparison_heatmap.png")
        
        # Bar plot
        plt.figure(figsize=(12, 8))
        
        # R² skorları için bar plot
        r2_data = self.comparison_df[self.comparison_df['R²'].notna()]
        
        if not r2_data.empty:
            sns.barplot(data=r2_data, x='Varlık', y='R²', hue='Model')
            plt.title('R² Skorları Karşılaştırması')
            plt.xticks(rotation=45)
            plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig('data/r2_comparison.png', dpi=300, bbox_inches='tight')
            print("✅ R² karşılaştırması kaydedildi: data/r2_comparison.png")
    
    def generate_report(self):
        """Rapor oluştur"""
        print("\n📄 Rapor oluşturuluyor...")
        
        report = []
        report.append("# Model Performans Karşılaştırma Raporu")
        report.append("="*50)
        report.append("")
        
        # En iyi model her varlık için
        for varlik in ['dolar_skor', 'altin_skor', 'borsa_skor', 'bitcoin_skor']:
            varlik_data = self.comparison_df[self.comparison_df['Varlık'] == varlik]
            
            if not varlik_data.empty:
                best_model = varlik_data.loc[varlik_data['R²'].idxmax()]
                report.append(f"## {varlik.replace('_skor', '').title()}")
                report.append(f"**En İyi Model:** {best_model['Model']}")
                report.append(f"**R² Skoru:** {best_model['R²']:.4f}")
                report.append(f"**MAE:** {best_model['MAE']:.4f}")
                report.append(f"**RMSE:** {best_model['RMSE']:.4f}")
                report.append("")
        
        # Genel özet
        report.append("## Genel Özet")
        overall_best = self.comparison_df.loc[self.comparison_df['R²'].idxmax()]
        report.append(f"**En İyi Genel Performans:** {overall_best['Model']} - {overall_best['Varlık']}")
        report.append(f"**R² Skoru:** {overall_best['R²']:.4f}")
        report.append("")
        
        # Model bazında ortalama performans
        report.append("## Model Bazında Ortalama Performans")
        model_avg = self.comparison_df.groupby('Model')['R²'].mean().sort_values(ascending=False)
        for model, avg_r2 in model_avg.items():
            report.append(f"- **{model}:** R² = {avg_r2:.4f}")
        
        # Raporu kaydet
        with open('data/model_comparison_report.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print("✅ Rapor kaydedildi: data/model_comparison_report.md")
    
    def run_comparison(self):
        """Tüm karşılaştırma sürecini çalıştır"""
        print("🎯 Model Karşılaştırma Süreci Başlıyor...")
        print("="*60)
        
        # 1. Test verilerini yükle
        if not self.load_test_data():
            print("❌ Test verileri yüklenemedi!")
            return
        
        # 2. Modelleri değerlendir
        self.evaluate_models()
        
        # 3. Karşılaştırma tablosu oluştur
        self.create_comparison_table()
        
        # 4. Görselleştirmeler oluştur
        self.create_visualizations()
        
        # 5. Rapor oluştur
        self.generate_report()
        
        print("\n🎉 Model karşılaştırması tamamlandı!")
        print("📁 Sonuçlar:")
        print("- data/model_comparison.xlsx")
        print("- data/model_comparison_heatmap.png")
        print("- data/r2_comparison.png")
        print("- data/model_comparison_report.md")

if __name__ == "__main__":
    # Karşılaştırmayı başlat
    comparator = ModelComparator()
    comparator.run_comparison() 